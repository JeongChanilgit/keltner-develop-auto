
import random
import copy
import time
import pandas as pd
import numpy as np
from backtest import load_data_grouped, BTConfig, BacktestEngine, save_experiment_result
from signal_engine import SignalConfig

# ==========================================
# Configuration Space
# ==========================================
PARAM_RANGES = {
    # ATR Risk Management
    "SL_ATR_MULT": (3.0, 15.0),    # Stop Loss multiplier
    "TP_ATR_MULT": (2.0, 15.0),    # Take Profit multiplier
    
    # Keltner Channel
    "KC_PERIOD": (10, 40),
    "KC_MULT": (1.0, 3.0),
    
    # Filters
    "ADX_THRESHOLD": (10, 30),
    "VOL_RATIO_MAX": (0.5, 3.0),
    "EMA_DIST_MIN": (0.001, 0.01),
    "EMA_DIST_MAX": (0.02, 0.05),
    "ATR_PCT_MIN": (0.001, 0.006),
    "ATR_PCT_MAX": (0.008, 0.02),
}

# Parameters that must be integers
INT_PARAMS = ["KC_PERIOD", "ADX_THRESHOLD"]

def get_random_params():
    params = {}
    for key, (mn, mx) in PARAM_RANGES.items():
        val = random.uniform(mn, mx)
        if key in INT_PARAMS:
            val = int(round(val))
        params[key] = val
    return params

def mutate_params(current_params, mutation_rate=0.3):
    new_params = current_params.copy()
    for key, (mn, mx) in PARAM_RANGES.items():
        # Mutate with probability
        if random.random() < mutation_rate:
            change_pct = random.uniform(-0.2, 0.2) # +/- 20% variation
            val = new_params[key]
            
            # Add variation
            new_val = val * (1 + change_pct)
            
            # Clip to range
            new_val = max(mn, min(mx, new_val))
            
            if key in INT_PARAMS:
                new_val = int(round(new_val))
                
            new_params[key] = new_val
    return new_params

def evaluate_score(results):
    total = results.get("Total", {})
    if not total:
        return -999999
        
    net_profit = total.get("NetProfit", 0)
    mdd = total.get("MDD", 100)
    count = total.get("Count", 0)
    win_rate = total.get("WinRate", 0)
    
    # Constraints / Penalties
    if count < 10: # Too few trades
        return -999999
    if mdd > 20.0: # Too risky
        return -999999
        
    # Score function: Profit heavily weighted, but penalized by MDD
    # Simple: Profit / MDD (Calmar-like)
    score = net_profit
    
    # Penalize low win rate?
    if win_rate < 30:
        score *= 0.5
        
    return score

def run_auto_optimizer(generations=50):
    print("📂 Loading Data...")
    grouped = load_data_grouped(BTConfig.DATA_DIR, BTConfig.TIMEFRAME)
    if not grouped:
        print("❌ Data load failed")
        return
        
    data = grouped[BTConfig.TIMEFRAME]
    btc_df = data["btc"]
    alt_dict = data["alts"]
    
    print(f"🚀 Starting Auto-Optimizer for {generations} iterations...")
    
    # Start with default config values
    best_params = {}
    for k in PARAM_RANGES.keys():
        if hasattr(SignalConfig, k):
            best_params[k] = getattr(SignalConfig, k)
        else:
            # Fallback if config misses it, take mid range
            mn, mx = PARAM_RANGES[k]
            best_params[k] = (mn + mx) / 2
            
    best_score = -float('inf')
    
    for gen in range(1, generations + 1):
        print(f"\n🧪 [Gen {gen}/{generations}] ", end="")
        
        # Strategy: Hill Climbing
        # 30% chance to purely random explore, 70% mutate best
        if random.random() < 0.3 and gen > 1:
            test_params = get_random_params()
            mode = "Exploration 🌍"
        else:
            test_params = mutate_params(best_params)
            mode = "Mutation 🧬"
            
        print(f"Mode: {mode}")
        
        # Construct Config Class
        class CurrentConfig(SignalConfig):
            pass
        for k, v in test_params.items():
            setattr(CurrentConfig, k, v)
            
        # Run
        try:
            engine = BacktestEngine(BTConfig.INITIAL_BALANCE)
            engine.run(btc_df, alt_dict, signal_config=CurrentConfig)
            results = engine.get_summary_by_side()
            
            # Score
            score = evaluate_score(results)
            total_res = results.get("Total", {})
            profit = total_res.get("NetProfit", 0)
            mdd = total_res.get("MDD", 0)
            count = total_res.get("Count", 0)
            
            print(f"   📊 Profit: ${profit:,.0f} | MDD: {mdd:.2f}% | Trades: {count} | Score: {score:.2f}")
            
            # Save Record
            save_experiment_result(results, CurrentConfig, engine=engine)
            
            # Update Best
            if score > best_score:
                print(f"   🎉 NEW BEST FOUND! (Prev: {best_score:.2f} -> New: {score:.2f})")
                best_score = score
                best_params = test_params
                
                # Write best params to a unique file to allow parallel execution (Swarm)
                safe_filename = f"best_params_score_{score:.0f}_{int(time.time())}.txt"
                with open(safe_filename, "w") as f:
                    f.write(str(best_params))
                print(f"   💾 Saved best params to: {safe_filename}")
                    
        except Exception as e:
            print(f"   ❌ Error in run: {e}")
            
    print("\n✅ Optimization Loop Finished.")
    print("🏆 Best Parameters:", best_params)

if __name__ == "__main__":
    run_auto_optimizer(generations=1000)
