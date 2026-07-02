param(
  [string]$RosWildfireSrc = "D:\jk2347\Workspace\ros-based-wildfire-prediction\src"
)

$ErrorActionPreference = "Stop"

$common = @(
  "--dataset-root", "dataset\pkl_materialized",
  "--max-events-per-scenario", "3",
  "--train-events-per-scenario", "2",
  "--max-frame-pairs-per-event", "5",
  "--warm-start", "pso",
  "--pso-device", "cpu",
  "--pso-t-cap", "48",
  "--ros-wildfire-src", $RosWildfireSrc,
  "--pso-cache-dir", "outputs\pso_cache",
  "--reasoner", "ollama",
  "--ollama-timeout-seconds", "900",
  "--selector", "area_budget"
)

python -m wildfire_llm_agent.cli calibrate-materialized-root @common --max-scenarios 5 --output-dir outputs\pixel_calibration_llama4_prompt_v2_5scenario_5frames --ollama-model llama4:latest --ollama-cache-dir outputs\ollama_cache_llama4_prompt_v2_large_update
python -m wildfire_llm_agent.cli calibrate-materialized-root @common --max-scenarios 5 --output-dir outputs\pixel_calibration_gemma4_31b_prompt_v2_5scenario_5frames --ollama-model gemma4:31b --ollama-cache-dir outputs\ollama_cache_gemma4_31b_prompt_v2_large_update

python -m wildfire_llm_agent.cli calibrate-materialized-root @common --max-scenarios 10 --output-dir outputs\pixel_calibration_llama4_prompt_v2_10scenario_5frames --ollama-model llama4:latest --ollama-cache-dir outputs\ollama_cache_llama4_prompt_v2_large_update
python -m wildfire_llm_agent.cli calibrate-materialized-root @common --max-scenarios 10 --output-dir outputs\pixel_calibration_gemma4_31b_prompt_v2_10scenario_5frames --ollama-model gemma4:31b --ollama-cache-dir outputs\ollama_cache_gemma4_31b_prompt_v2_large_update

$cvCommon = @(
  "--dataset-root", "dataset\pkl_materialized",
  "--max-scenarios", "5",
  "--max-events-per-scenario", "3",
  "--max-frame-pairs-per-event", "5",
  "--warm-start", "pso",
  "--pso-device", "cpu",
  "--pso-t-cap", "48",
  "--ros-wildfire-src", $RosWildfireSrc,
  "--pso-cache-dir", "outputs\pso_cache",
  "--reasoner", "ollama",
  "--ollama-timeout-seconds", "900",
  "--selector", "area_budget"
)

python -m wildfire_llm_agent.cli cross-validate-materialized-calibration @cvCommon --output-dir outputs\pixel_calibration_llama4_prompt_v2_rotating_cv_5scenario_5frames --ollama-model llama4:latest --ollama-cache-dir outputs\ollama_cache_llama4_prompt_v2_large_update
python -m wildfire_llm_agent.cli cross-validate-materialized-calibration @cvCommon --output-dir outputs\pixel_calibration_gemma4_31b_prompt_v2_rotating_cv_5scenario_5frames --ollama-model gemma4:31b --ollama-cache-dir outputs\ollama_cache_gemma4_31b_prompt_v2_large_update

python scripts\compare_model_runs.py --run llama4=outputs\pixel_calibration_llama4_prompt_v2_5scenario_5frames --run gemma4_31b=outputs\pixel_calibration_gemma4_31b_prompt_v2_5scenario_5frames --output-csv outputs\prompt_v2_model_comparison_5scenario.csv --output-md docs\prompt_v2_model_comparison_5scenario.md
python scripts\compare_model_runs.py --run llama4=outputs\pixel_calibration_llama4_prompt_v2_10scenario_5frames --run gemma4_31b=outputs\pixel_calibration_gemma4_31b_prompt_v2_10scenario_5frames --output-csv outputs\prompt_v2_model_comparison_10scenario.csv --output-md docs\prompt_v2_model_comparison_10scenario.md
python scripts\compare_model_runs.py --run llama4=outputs\pixel_calibration_llama4_prompt_v2_rotating_cv_5scenario_5frames --run gemma4_31b=outputs\pixel_calibration_gemma4_31b_prompt_v2_rotating_cv_5scenario_5frames --output-csv outputs\prompt_v2_model_comparison_rotating_cv.csv --output-md docs\prompt_v2_model_comparison_rotating_cv.md
