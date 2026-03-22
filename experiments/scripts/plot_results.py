import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
 
 
def create_plots(csv_file, plots_dir='.'):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: '{csv_file}' not found.")
        return
 
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
 
    df['device_precision'] = df['device'] + ' ' + df['precision']
    batch_sizes = sorted(df['batch_size'].unique())
 
    # --- Plot 1: Throughput vs Batch Size ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='batch_size', y='throughput_fps',
                 hue='device', style='precision', marker='o')
    plt.title('Throughput vs. Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (Images/Second)')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=batch_sizes)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/01_throughput_vs_batch_size.png")
    print("Saved 01_throughput_vs_batch_size.png")
    plt.close()
 
    # --- Plot 2: Latency vs Batch Size ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='batch_size', y='avg_latency_ms',
                 hue='device', style='precision', marker='o')
    plt.title('Average Latency vs. Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=batch_sizes)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/02_latency_vs_batch_size.png")
    print("Saved 02_latency_vs_batch_size.png")
    plt.close()
 
    # --- Plot 3: Recall@5 by device and precision ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='device', y='recall_at_5', hue='precision')
    plt.title('Recall@5 by Device and Precision')
    plt.xlabel('Device')
    plt.ylabel('Recall@5')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/03_recall_at_5.png")
    print("Saved 03_recall_at_5.png")
    plt.close()
 
    # --- Plot 4: GPU Peak Memory vs Batch Size ---
    df_gpu = df[df['device'] != 'CPU']
    if not df_gpu.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_gpu, x='batch_size', y='max_memory_mb',
                     hue='precision', marker='o')
        plt.title('Peak GPU Memory vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Peak Memory (MB)')
        plt.xscale('log', base=2)
        plt.xticks(batch_sizes, labels=batch_sizes)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/04_memory_vs_batch_size.png")
        print("Saved 04_memory_vs_batch_size.png")
        plt.close()
 
    # --- Plot 5: Power Draw vs Batch Size (GPU only) ---
    power_col = 'net_compute_power_mw' if 'net_compute_power_mw' in df.columns else 'avg_power_mw'
    if not df_gpu.empty and df_gpu[power_col].sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_gpu, x='batch_size', y=power_col,
                     hue='precision', marker='o')
        plt.title('Net Compute Power vs. Batch Size (Jetson GPU)')
        plt.xlabel('Batch Size')
        plt.ylabel('Power (mW)')
        plt.xscale('log', base=2)
        plt.xticks(batch_sizes, labels=batch_sizes)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/05_power_vs_batch_size.png")
        print("Saved 05_power_vs_batch_size.png")
        plt.close()
 
    # --- Plot 6: FP16 Speedup over FP32 (GPU) ---
    if not df_gpu.empty:
        fp32 = df_gpu[df_gpu['precision'] == 'FP32'].set_index('batch_size')['avg_latency_ms']
        fp16 = df_gpu[df_gpu['precision'] == 'FP16'].set_index('batch_size')['avg_latency_ms']
        common = fp32.index.intersection(fp16.index)
        speedup = (fp32[common] / fp16[common]).reset_index()
        speedup.columns = ['batch_size', 'speedup']
 
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=speedup, x='batch_size', y='speedup', color='steelblue')
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}x", ha='center', va='bottom', fontsize=10)
        plt.axhline(y=1.0, color='red', linestyle='--', label='No speedup (1.0x)')
        plt.title('FP16 Speedup over FP32 (GPU)')
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup (FP32 latency / FP16 latency)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/06_fp16_speedup.png")
        print("Saved 06_fp16_speedup.png")
        plt.close()
 
    # --- Plot 7: FP16 vs FP32 Comparison (Latency, Memory, Power side by side) ---
    if not df_gpu.empty:
        df_avg = df_gpu.groupby('precision')[['avg_latency_ms', 'max_memory_mb', power_col]].mean().reset_index()
        colors = ['#4878CF', '#6ACC65']
 
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('FP16 vs FP32 Comparison (GPU, averaged across batch sizes)', fontsize=13)
 
        sns.barplot(data=df_avg, x='precision', y='avg_latency_ms', ax=axes[0], palette=colors)
        axes[0].set_title('Average Latency')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_xlabel('Precision')
        for bar in axes[0].patches:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=10)
 
        sns.barplot(data=df_avg, x='precision', y='max_memory_mb', ax=axes[1], palette=colors)
        axes[1].set_title('Peak GPU Memory')
        axes[1].set_ylabel('Memory (MB)')
        axes[1].set_xlabel('Precision')
        for bar in axes[1].patches:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=10)
 
        if df_avg[power_col].sum() > 0:
            sns.barplot(data=df_avg, x='precision', y=power_col, ax=axes[2], palette=colors)
            axes[2].set_title('Net Compute Power')
            axes[2].set_ylabel('Power (mW)')
            axes[2].set_xlabel('Precision')
            for bar in axes[2].patches:
                axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                             f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=10)
        else:
            axes[2].set_visible(False)
 
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/07_fp16_vs_fp32_comparison.png")
        print("Saved 07_fp16_vs_fp32_comparison.png")
        plt.close()
 
    # --- Plot 8: Energy per Image (GPU only, if available) ---
    if not df_gpu.empty and 'energy_per_image_mj' in df.columns and df_gpu['energy_per_image_mj'].sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_gpu, x='batch_size', y='energy_per_image_mj',
                     hue='precision', marker='o')
        plt.title('Energy per Image vs. Batch Size (Jetson GPU)')
        plt.xlabel('Batch Size')
        plt.ylabel('Energy per Image (mJ)')
        plt.xscale('log', base=2)
        plt.xticks(batch_sizes, labels=batch_sizes)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/08_energy_per_image.png")
        print("Saved 08_energy_per_image.png")
        plt.close()
 
    print(f"\nAll plots saved to '{plots_dir}/'")
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot results from CLIP experiments.")
    parser.add_argument('--csv_file', type=str, default='../results/results.csv')
    parser.add_argument('--plots_dir', type=str, default='../results/plots')
    args = parser.parse_args()
    create_plots(args.csv_file, args.plots_dir)
