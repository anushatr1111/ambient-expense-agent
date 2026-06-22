# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load API key and project environment variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

# Add google-agents-cli site-packages to sys.path
sys.path.insert(0, r"C:\Users\Lenovo\AppData\Roaming\uv\tools\google-agents-cli\Lib\site-packages")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Mock GCP credentials/auth/storage to make it 100% offline and local friendly
mock_creds = MagicMock()
patch("google.auth.default", return_value=(mock_creds, "mock-project")).start()
patch("google.cloud.storage.Client").start()

import vertexai
from google.agents.cli.eval.eval_utils import prepare_eval_metrics, print_results_table
from vertexai._genai.types.common import EvaluationDataset
from rich.console import Console

def main():
    console = Console()
    
    traces_path = "artifacts/traces/generated_traces.json"
    config_path = "tests/eval/eval_config.yaml"
    
    if not os.path.exists(traces_path):
        console.print(f"[bold red]Error:[/] Traces file not found at {traces_path}. Please run generate-traces first.")
        sys.exit(1)
        
    if not os.path.exists(config_path):
        console.print(f"[bold red]Error:[/] Config file not found at {config_path}.")
        sys.exit(1)
        
    console.print(f"[bold green]Loading evaluation traces...[/]")
    with open(traces_path, "r", encoding="utf-8") as f:
        dataset = EvaluationDataset.model_validate_json(f.read())
        
    console.print(f"[bold green]Preparing evaluation metrics...[/]")
    metrics, _, _ = prepare_eval_metrics(config_path=config_path, metrics_str=None)
    
    console.print(f"[bold green]Initializing evaluation client and executing evaluations...[/]")
    client = vertexai.Client(project="mock-project", location="global")
    
    try:
        result = client.evals.evaluate(dataset=dataset, metrics=metrics)
        console.print("\n[bold green]Evaluation grading complete![/]\n")
        
        # Display the aggregate summary table
        print_results_table(result, console)
        
        # Detailed case explanations
        console.print("\n[bold cyan]Detailed Evaluation Cases & Explanations:[/]")
        eval_cases = result.evaluation_dataset[0].eval_cases
        
        for case_result in result.eval_case_results:
            idx = case_result.eval_case_index
            eval_case = eval_cases[idx]
            console.print(f"\n[bold yellow]======================================[/]")
            console.print(f"[bold white]Eval Case ID:[/] [cyan]{eval_case.eval_case_id}[/]")
            
            if case_result.response_candidate_results:
                candidate_res = case_result.response_candidate_results[0]
                for metric_name, metric_res in candidate_res.metric_results.items():
                    console.print(f"[bold blue]--- Metric: {metric_name} ---[/]")
                    if metric_res.error_message:
                        console.print(f"[bold red]Status:[/] Error: {metric_res.error_message}")
                    else:
                        score_color = "green" if metric_res.score >= 4.0 else "yellow" if metric_res.score >= 3.0 else "red"
                        console.print(f"[bold]Score:[/] [{score_color}]{metric_res.score}[/]")
                        console.print(f"[bold]Explanation:[/] {metric_res.explanation}")
            else:
                console.print("[red]No response candidate results returned.[/]")
                
    except Exception as e:
        console.print(f"[bold red]Evaluation failed to run:[/] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
