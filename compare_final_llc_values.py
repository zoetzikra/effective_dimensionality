#!/usr/bin/env python3
"""
Compare Final LLC Values Across Models and Training Methods

This script extracts the final LLC values from trajectory data and creates
visualizations to compare adversarially robust vs vulnerable models.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

class LLCComparisonAnalyzer:
    """Analyze and compare final LLC values across different models and training methods"""
    
    def __init__(self, llc_analysis_dir: str = "./llc_analysis"):
        self.llc_analysis_dir = Path(llc_analysis_dir)
        self.results = []
        
    def extract_final_llc_values(self) -> List[Dict]:
        """Extract final LLC values from all available trajectory data"""
        results = []
        
        # Define robustness categories
        robust_methods = ['AT', 'TRADES', 'MART', 'AT_AWP', 'ATAWP']
        vulnerable_methods = ['Standard']
        
        # Search for comparison files in organized directories (exclude timestamped ones)
        for model_dir in self.llc_analysis_dir.glob("llc_analysis_*"):
            if not model_dir.is_dir():
                continue
            
            # Skip timestamped directories (format: llc_analysis_YYYYMMDD_HHMMSS)
            dir_name = model_dir.name
            if len(dir_name.split('_')) >= 3 and dir_name.split('_')[2].isdigit():
                continue  # Skip timestamped directories
                
            # Look for clean_vs_adv_comparison.json files
            comparison_files = list(model_dir.glob("**/clean_vs_adv_comparison.json"))
            
            for comp_file in comparison_files:
                try:
                    with open(comp_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract basic info
                    model_name = data.get('model_name', 'Unknown')
                    dataset = data.get('dataset_name', 'Unknown')
                    defense_method = data.get('defense_method', 'Unknown')
                    
                    # Determine robustness category
                    if defense_method in robust_methods:
                        robustness_category = 'Robust'
                    elif defense_method in vulnerable_methods:
                        robustness_category = 'Vulnerable'
                    else:
                        robustness_category = 'Unknown'
                    
                    # Extract final LLC values from trajectories
                    clean_final_llc = None
                    adv_final_llc = None
                    
                    # Clean trajectory
                    if 'clean_trajectory' in data and 'llc_means' in data['clean_trajectory']:
                        clean_means = data['clean_trajectory']['llc_means']
                        if clean_means and len(clean_means) > 0:
                            clean_final_llc = clean_means[-1]  # Last value
                    
                    # Adversarial trajectory
                    if 'adversarial_trajectory' in data and 'llc_means' in data['adversarial_trajectory']:
                        adv_means = data['adversarial_trajectory']['llc_means']
                        if adv_means and len(adv_means) > 0:
                            adv_final_llc = adv_means[-1]  # Last value
                    
                    # Also check optimal_params for single LLC values
                    if 'optimal_params' in data and 'llc_mean' in data['optimal_params']:
                        single_llc = data['optimal_params']['llc_mean']
                        if clean_final_llc is None:
                            clean_final_llc = single_llc
                    
                    # Create result entry
                    result = {
                        'model_name': model_name,
                        'dataset': dataset,
                        'defense_method': defense_method,
                        'robustness_category': robustness_category,
                        'clean_final_llc': clean_final_llc,
                        'adversarial_final_llc': adv_final_llc,
                        'source_file': str(comp_file)
                    }
                    
                    results.append(result)
                    llc_display = f"{clean_final_llc:.4f}" if clean_final_llc is not None else "N/A"
                    adv_llc_display = f"{adv_final_llc:.4f}" if adv_final_llc is not None else "N/A"
                    print(f"Extracted: {model_name}-{defense_method} (Clean LLC: {llc_display}, Adversarial LLC: {adv_llc_display})")
                    
                except Exception as e:
                    print(f"Error processing {comp_file}: {e}")
                    continue
        
        self.results = results
        return results
    
    def create_comparison_visualizations(self, save_dir: str = "./llc_comparison_plots"):
        """Create multiple visualizations to compare LLC values"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to plot. Run extract_final_llc_values() first.")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results)
        
        # Filter out entries without clean LLC values
        df_clean = df[df['clean_final_llc'].notna()].copy()
        
        if len(df_clean) == 0:
            print("No valid clean LLC values found.")
            return
        
        print(f"Creating visualizations with {len(df_clean)} models...")
        
        # 1. Bar plot comparing robust vs vulnerable models
        self._create_robustness_comparison_plot(df_clean, save_dir)
        
        # 2. Scatter plot: Clean vs Adversarial LLC
        self._create_clean_vs_adversarial_scatter(df_clean, save_dir)
        
        # 3. Box plot by defense method
        self._create_defense_method_boxplot(df_clean, save_dir)
        
        # 4. Heatmap by model and defense method
        self._create_model_defense_heatmap(df_clean, save_dir)
        
        # 5. Architecture-specific robustness comparison (excluding AWP)
        self._create_architecture_robustness_comparison(df_clean, save_dir)
        
        # 6. AWP-specific comparison
        self._create_awp_comparison_plot(df_clean, save_dir)
        
        # 7. Summary statistics
        self._create_summary_statistics(df_clean, save_dir)
        
        print(f"All visualizations saved to: {save_dir}")
    
    def _create_robustness_comparison_plot(self, df: pd.DataFrame, save_dir: Path):
        """Create bar plot comparing robust vs vulnerable models"""
        plt.figure(figsize=(12, 8))
        
        # Group by robustness category
        robust_data = df[df['robustness_category'] == 'Robust']['clean_final_llc']
        vulnerable_data = df[df['robustness_category'] == 'Vulnerable']['clean_final_llc']
        
        # Create grouped bar plot
        categories = []
        means = []
        stds = []
        colors = []
        
        if len(robust_data) > 0:
            categories.append('Adversarially\nRobust')
            means.append(robust_data.mean())
            stds.append(robust_data.std())
            colors.append('darkgreen')
        
        if len(vulnerable_data) > 0:
            categories.append('Adversarially\nVulnerable')
            means.append(vulnerable_data.mean())
            stds.append(vulnerable_data.std())
            colors.append('darkred')
        
        bars = plt.bar(categories, means, yerr=stds, capsize=10, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add individual data points
        if len(robust_data) > 0:
            x_robust = [0] * len(robust_data)
            plt.scatter(x_robust, robust_data, color='darkgreen', alpha=0.8, s=100, 
                       edgecolors='black', linewidth=1, zorder=3)
        
        if len(vulnerable_data) > 0:
            x_vulnerable = [1] * len(vulnerable_data) if len(categories) > 1 else [0] * len(vulnerable_data)
            plt.scatter(x_vulnerable, vulnerable_data, color='darkred', alpha=0.8, s=100,
                       edgecolors='black', linewidth=1, zorder=3)
        
        plt.ylabel('Final LLC Value', fontsize=14)
        plt.title('Final LLC Values: Adversarially Robust vs Vulnerable Models', 
                 fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'robustness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_clean_vs_adversarial_scatter(self, df: pd.DataFrame, save_dir: Path):
        """Create scatter plot of clean vs adversarial LLC values"""
        # Filter for entries with both clean and adversarial LLC
        df_both = df[(df['clean_final_llc'].notna()) & (df['adversarial_final_llc'].notna())].copy()
        
        if len(df_both) == 0:
            print("No models with both clean and adversarial LLC values found.")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Color by robustness category
        colors = {'Robust': 'darkgreen', 'Vulnerable': 'darkred', 'Unknown': 'gray'}
        
        for category in df_both['robustness_category'].unique():
            subset = df_both[df_both['robustness_category'] == category]
            plt.scatter(subset['clean_final_llc'], subset['adversarial_final_llc'],
                       c=colors.get(category, 'gray'), label=f'{category} Models',
                       s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add model labels
            for _, row in subset.iterrows():
                plt.annotate(f"{row['model_name']}-{row['defense_method']}", 
                           (row['clean_final_llc'], row['adversarial_final_llc']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add diagonal line for reference
        min_val = min(df_both['clean_final_llc'].min(), df_both['adversarial_final_llc'].min())
        max_val = max(df_both['clean_final_llc'].max(), df_both['adversarial_final_llc'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
        
        plt.xlabel('Clean Data Final LLC', fontsize=14)
        plt.ylabel('Adversarial Data Final LLC', fontsize=14)
        plt.title('Clean vs Adversarial Final LLC Values', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'clean_vs_adversarial_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_defense_method_boxplot(self, df: pd.DataFrame, save_dir: Path):
        """Create box plot by defense method"""
        plt.figure(figsize=(14, 8))
        
        # Create box plot
        defense_methods = df['defense_method'].unique()
        data_by_method = [df[df['defense_method'] == method]['clean_final_llc'].values 
                         for method in defense_methods]
        
        box_plot = plt.boxplot(data_by_method, labels=defense_methods, patch_artist=True)
        
        # Color boxes by robustness
        robust_methods = ['AT', 'TRADES', 'MART', 'AT_AWP', 'ATAWP']
        for i, method in enumerate(defense_methods):
            if method in robust_methods:
                box_plot['boxes'][i].set_facecolor('lightgreen')
            else:
                box_plot['boxes'][i].set_facecolor('lightcoral')
        
        plt.ylabel('Final LLC Value', fontsize=14)
        plt.xlabel('Defense Method', fontsize=14)
        plt.title('Final LLC Values by Defense Method', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'defense_method_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_defense_heatmap(self, df: pd.DataFrame, save_dir: Path):
        """Create heatmap of LLC values by model and defense method"""
        # Create pivot table
        pivot_table = df.pivot_table(values='clean_final_llc', 
                                   index='model_name', 
                                   columns='defense_method', 
                                   aggfunc='mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', 
                   cbar_kws={'label': 'Final LLC Value'})
        plt.title('Final LLC Values: Model Architecture vs Defense Method', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Defense Method', fontsize=14)
        plt.ylabel('Model Architecture', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'model_defense_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_architecture_robustness_comparison(self, df: pd.DataFrame, save_dir: Path):
        """Create bar plot comparing robust vs vulnerable for each architecture (excluding AWP)"""
        # Filter out AWP models
        df_no_awp = df[~df['defense_method'].str.contains('AWP', na=False)].copy()
        
        if len(df_no_awp) == 0:
            print("No non-AWP models found for architecture comparison.")
            return
        
        # Get unique architectures
        architectures = df_no_awp['model_name'].unique()
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Set up bar positions
        x = np.arange(len(architectures))
        width = 0.35
        
        robust_means = []
        robust_stds = []
        vulnerable_means = []
        vulnerable_stds = []
        
        for arch in architectures:
            arch_data = df_no_awp[df_no_awp['model_name'] == arch]
            
            # Robust models for this architecture
            robust_data = arch_data[arch_data['robustness_category'] == 'Robust']['clean_final_llc']
            if len(robust_data) > 0:
                robust_means.append(robust_data.mean())
                robust_stds.append(robust_data.std() if len(robust_data) > 1 else 0)
            else:
                robust_means.append(0)
                robust_stds.append(0)
            
            # Vulnerable models for this architecture
            vulnerable_data = arch_data[arch_data['robustness_category'] == 'Vulnerable']['clean_final_llc']
            if len(vulnerable_data) > 0:
                vulnerable_means.append(vulnerable_data.mean())
                vulnerable_stds.append(vulnerable_data.std() if len(vulnerable_data) > 1 else 0)
            else:
                vulnerable_means.append(0)
                vulnerable_stds.append(0)
        
        # Create bars
        bars1 = ax.bar(x - width/2, vulnerable_means, width, yerr=vulnerable_stds, 
                      label='Vulnerable (Standard)', color='darkred', alpha=0.7, 
                      capsize=5, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, robust_means, width, yerr=robust_stds,
                      label='Robust (AT/TRADES/MART)', color='darkgreen', alpha=0.7,
                      capsize=5, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bars, means, stds in [(bars1, vulnerable_means, vulnerable_stds), 
                                 (bars2, robust_means, robust_stds)]:
            for bar, mean, std in zip(bars, means, stds):
                if mean > 0:  # Only label non-zero bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                           f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model Architecture', fontsize=14)
        ax.set_ylabel('Final LLC Value', fontsize=14)
        ax.set_title('LLC Values by Architecture: Robust vs Vulnerable Models\n(Excluding AWP variants)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(architectures)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'architecture_robustness_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'architecture_robustness_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def _create_awp_comparison_plot(self, df: pd.DataFrame, save_dir: Path):
        """Create bar plot comparing Standard vs Robust vs Robust+AWP models"""
        # Get models that have AWP variants
        awp_models = df[df['defense_method'].str.contains('AWP', na=False)]
        
        if len(awp_models) == 0:
            print("No AWP models found for AWP comparison.")
            return
        
        # Get the base architectures that have AWP variants
        awp_architectures = awp_models['model_name'].unique()
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Set up bar positions
        x = np.arange(len(awp_architectures))
        width = 0.25
        
        standard_means = []
        standard_stds = []
        robust_means = []
        robust_stds = []
        awp_means = []
        awp_stds = []
        
        for arch in awp_architectures:
            arch_data = df[df['model_name'] == arch]
            
            # Standard models for this architecture
            standard_data = arch_data[arch_data['defense_method'] == 'Standard']['clean_final_llc']
            if len(standard_data) > 0:
                standard_means.append(standard_data.mean())
                standard_stds.append(standard_data.std() if len(standard_data) > 1 else 0)
            else:
                standard_means.append(0)
                standard_stds.append(0)
            
            # Robust models (non-AWP) for this architecture
            robust_data = arch_data[
                (arch_data['robustness_category'] == 'Robust') & 
                (~arch_data['defense_method'].str.contains('AWP', na=False))
            ]['clean_final_llc']
            if len(robust_data) > 0:
                robust_means.append(robust_data.mean())
                robust_stds.append(robust_data.std() if len(robust_data) > 1 else 0)
            else:
                robust_means.append(0)
                robust_stds.append(0)
            
            # AWP models for this architecture
            awp_data = arch_data[arch_data['defense_method'].str.contains('AWP', na=False)]['clean_final_llc']
            if len(awp_data) > 0:
                awp_means.append(awp_data.mean())
                awp_stds.append(awp_data.std() if len(awp_data) > 1 else 0)
            else:
                awp_means.append(0)
                awp_stds.append(0)
        
        # Create bars
        bars1 = ax.bar(x - width, standard_means, width, yerr=standard_stds,
                      label='Standard Training', color='darkred', alpha=0.7,
                      capsize=5, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, robust_means, width, yerr=robust_stds,
                      label='Robust Training', color='darkgreen', alpha=0.7,
                      capsize=5, edgecolor='black', linewidth=1)
        bars3 = ax.bar(x + width, awp_means, width, yerr=awp_stds,
                      label='Robust + AWP', color='darkblue', alpha=0.7,
                      capsize=5, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bars, means, stds in [(bars1, standard_means, standard_stds),
                                 (bars2, robust_means, robust_stds),
                                 (bars3, awp_means, awp_stds)]:
            for bar, mean, std in zip(bars, means, stds):
                if mean > 0:  # Only label non-zero bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                           f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model Architecture', fontsize=14)
        ax.set_ylabel('Final LLC Value', fontsize=14)
        ax.set_title('LLC Values: Standard vs Robust vs Robust+AWP Training', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(awp_architectures)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'awp_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'awp_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def _create_summary_statistics(self, df: pd.DataFrame, save_dir: Path):
        """Create and save summary statistics"""
        summary = {}
        
        # Overall statistics
        summary['overall'] = {
            'total_models': len(df),
            'mean_llc': df['clean_final_llc'].mean(),
            'std_llc': df['clean_final_llc'].std(),
            'min_llc': df['clean_final_llc'].min(),
            'max_llc': df['clean_final_llc'].max()
        }
        
        # By robustness category
        summary['by_robustness'] = {}
        for category in df['robustness_category'].unique():
            subset = df[df['robustness_category'] == category]['clean_final_llc']
            summary['by_robustness'][category] = {
                'count': len(subset),
                'mean': subset.mean(),
                'std': subset.std(),
                'min': subset.min(),
                'max': subset.max()
            }
        
        # By defense method
        summary['by_defense_method'] = {}
        for method in df['defense_method'].unique():
            subset = df[df['defense_method'] == method]['clean_final_llc']
            summary['by_defense_method'][method] = {
                'count': len(subset),
                'mean': subset.mean(),
                'std': subset.std(),
                'min': subset.min(),
                'max': subset.max()
            }
        
        # By model architecture
        summary['by_model'] = {}
        for model in df['model_name'].unique():
            subset = df[df['model_name'] == model]['clean_final_llc']
            summary['by_model'][model] = {
                'count': len(subset),
                'mean': subset.mean(),
                'std': subset.std(),
                'min': subset.min(),
                'max': subset.max()
            }
        
        # Save summary
        with open(save_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create readable summary report
        with open(save_dir / 'summary_report.txt', 'w') as f:
            f.write("LLC COMPARISON ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Models Analyzed: {summary['overall']['total_models']}\n")
            f.write(f"Overall Mean LLC: {summary['overall']['mean_llc']:.4f} ± {summary['overall']['std_llc']:.4f}\n")
            f.write(f"LLC Range: {summary['overall']['min_llc']:.4f} - {summary['overall']['max_llc']:.4f}\n\n")
            
            f.write("BY ROBUSTNESS CATEGORY:\n")
            f.write("-" * 30 + "\n")
            for category, stats in summary['by_robustness'].items():
                f.write(f"{category}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
            
            f.write("\nBY DEFENSE METHOD:\n")
            f.write("-" * 30 + "\n")
            for method, stats in summary['by_defense_method'].items():
                f.write(f"{method}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
            
            f.write("\nBY MODEL ARCHITECTURE:\n")
            f.write("-" * 30 + "\n")
            for model, stats in summary['by_model'].items():
                f.write(f"{model}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
        
        print(f"Summary statistics saved to: {save_dir}")

def main():
    """Main function to run the LLC comparison analysis"""
    analyzer = LLCComparisonAnalyzer()
    
    print("Extracting final LLC values from trajectory data...")
    results = analyzer.extract_final_llc_values()
    
    if not results:
        print("No LLC data found. Please check your llc_analysis directory.")
        return
    
    print(f"\nFound {len(results)} model configurations.")
    
    print("\nCreating comparison visualizations...")
    analyzer.create_comparison_visualizations()
    
    print("\nAnalysis complete! Check the llc_comparison_plots directory for results.")

if __name__ == "__main__":
    main()
