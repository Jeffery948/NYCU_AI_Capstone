import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot(args):
    env_name = args.env_name
    seeds = [0, 4, 9]

    # 儲存多種設定的 DataFrame
    if env_name == 'LunarLander-v3':
         settings = {
            "Vanilla": [],
            "RewardNorm": [],
            "Baseline": [],
            "GAE": [],
            "EntropyReg": [],
            "RewardNorm_Baseline_EntropyReg": [],
            "RewardNorm_GAE_EntropyReg": []
        }
    else:
        settings = {
            "Vanilla": [],
            "RewardClip": [],
            "Double": [],
            "Dueling": [],
            "PER": [],
            "Double_Dueling_PER": []
        }

    # 讀檔案進來
    for seed in seeds:
        for setting in settings:
            path = f"logs/{setting}/{env_name}_seed{seed}_eval_log.csv"
            df = pd.read_csv(path)
            settings[setting].append(df)

    # 合併、平均每個 setting
    mean_dfs = {}
    for setting in settings:
        combined = pd.concat(settings[setting])
        mean_df = combined.groupby("episode")["eval_return"].agg(['mean', 'std']).reset_index()
        mean_dfs[setting] = mean_df

    # 設定 random policy 的表現
    random_mean = -170.15 if env_name == 'LunarLander-v3' else 238.00

    # 第一張圖：個別技巧比較
    plt.figure(figsize=(10, 6))
    if env_name == 'LunarLander-v3':
        individual = ["Vanilla", "RewardNorm", "Baseline", "GAE", "EntropyReg"]
    else:
        individual = ["Vanilla", "RewardClip", "Double", "Dueling", "PER"]

    for setting in individual:
        df = mean_dfs[setting]
        plt.plot(df['episode'], df['mean'], label=setting)
        plt.fill_between(df['episode'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.2)

    # Random Policy 虛線參考線
    episodes = mean_dfs["Vanilla"]['episode']
    plt.plot(episodes, [random_mean]*len(episodes), '--', color='gray', label='Random Policy')

    plt.xlabel("Episode", fontsize=22)
    plt.ylabel("Evaluation Return", fontsize=22)
    plt.title(f"Individual Enhancements on {env_name}", fontsize=22)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graphs/individual_enhancements_{env_name}.png", dpi=300)
    plt.show()

    # 第二張圖：完整組合比較
    plt.figure(figsize=(10, 6))
    if env_name == 'LunarLander-v3':
        combined = ["Vanilla", "RewardNorm_Baseline_EntropyReg", "RewardNorm_GAE_EntropyReg"]
    else:
        combined = ["Vanilla", "Double_Dueling_PER"]

    for setting in combined:
        df = mean_dfs[setting]
        plt.plot(df['episode'], df['mean'], label=setting)
        plt.fill_between(df['episode'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.2)

    # Random Policy 虛線參考線
    episodes = mean_dfs["Vanilla"]['episode']
    plt.plot(episodes, [random_mean]*len(episodes), '--', color='gray', label='Random Policy')

    plt.xlabel("Episode", fontsize=22)
    plt.ylabel("Evaluation Return", fontsize=22)
    plt.title(f"Complete Models on {env_name}", fontsize=22)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graphs/complete_models_{env_name}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default='LunarLander-v3')
	args = parser.parse_args()
	plot(args)