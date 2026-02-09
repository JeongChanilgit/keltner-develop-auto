
import logging
from backtest import load_data_grouped, BTConfig, BacktestEngine, save_experiment_result
from signal_engine_modified import SignalConfig

def run_optimization():
    # 1. Load Data once
    print("📂 Loading data for optimization...")
    grouped = load_data_grouped(BTConfig.DATA_DIR, BTConfig.TIMEFRAME)
    if not grouped:
        print("❌ No data found.")
        return
    data = grouped[BTConfig.TIMEFRAME]
    btc_df = data["btc"]
    alt_dict = data["alts"]

    # 2. Define Hyperparameter Space
    # Example: Tweaking Thresholds
    param_grid = [
        {"SL_ATR_MULT": 6, "TP_ATR_MULT": 6},
        {"SL_ATR_MULT": 8, "TP_ATR_MULT": 4},
        {"SL_ATR_MULT": 8, "TP_ATR_MULT": 8},
        {"SL_ATR_MULT": 10, "TP_ATR_MULT": 5}, # Default
        {"SL_ATR_MULT": 12, "TP_ATR_MULT": 6},
    ]

    print(f"🚀 Starting optimization loop with {len(param_grid)} variations...")

    for i, params in enumerate(param_grid):
        print(f"\n[Run {i+1}/{len(param_grid)}] Parameters: {params}")

        # Create a dynamic configuration class
        class CurrentConfig(SignalConfig):
            pass
        
        # Apply params
        for k, v in params.items():
            setattr(CurrentConfig, k, v)

        # Run Backtest
        engine = BacktestEngine(BTConfig.INITIAL_BALANCE)
        engine.run(btc_df, alt_dict, signal_config=CurrentConfig)
        
        # Get Results
        results = engine.get_summary_by_side()
        
        # Save to History
        # We pass CurrentConfig so the parameters are recorded correctly in the CSV
        save_experiment_result(results, CurrentConfig)
        
        total_res = results.get("Total", {})
        print(f"   👉 Result: NetProfit=${total_res.get('NetProfit',0):,.0f}, MDD={total_res.get('MDD',0):.2f}%, Count={total_res.get('Count',0)}")

    print("\n✅ Optimization Complete.")

if __name__ == "__main__":
    run_optimization()
