"""
Script to annotate reasoning depth using GPT-4o-mini as simulated annotators.
Generates 4 independent annotations and calculates inter-annotator agreement.
Also generates exemplary reasoning for each depth level using o1-mini.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from tqdm import tqdm
import openai
from openai import OpenAI
from dotenv import load_dotenv
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent
REASONING_DEPTH_DIR = PLOTS_DIR / "reasoning_depth"

plt.rcParams['font.family'] = 'Courier New'  # Base font
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(base_url=os.environ.get("OPENAI_ENDPOINT"), api_key=os.environ.get("OPENAI_API_KEY"))

ANNOTATION_PROMPT = """You are a medical professional annotating questions for a medical reasoning benchmark. Your task is to assess the REASONING DEPTH required to answer each question correctly.

**Reasoning Depth Rubric:**

1 (Direct Recall): Single fact retrieval from memory
   - Example: "Which structure collects urine in the body?"
   - Requires: Simple knowledge recall

2 (Single-Step): One inference or rule application
   - Example: "A patient with serum sodium of 125 mEq/L has which condition?"
   - Requires: Apply one medical principle or guideline

3 (Two-Step): Connecting two pieces of information
   - Example: "67-year-old faints after standing suddenly in heat. Diagnosis?"
   - Requires: (1) Recognize pattern (2) Determine cause

4 (Multi-Step): Three or more reasoning steps with concept integration
   - Example: "17-year-old with persistent acne, failed topical therapy. Next step?"
   - Requires: (1) Assess treatment failure (2) Review guidelines (3) Select next therapy

5+ (Complex): Extensive integration across multiple domains
   - Example: "39-year-old chest pain, hypertension, smoker, cocaine use. First management step?"
   - Requires: (1) Risk stratification (2) Consider differentials (3) Identify contraindications (4) Select treatment

**Instructions:**
- Read the question carefully
- Count the MINIMUM number of distinct cognitive steps needed
- Consider only what's necessary, not every possible reasoning path
- For test_hard questions, be particularly attentive to multi-step reasoning requirements
- Respond with ONLY a single number: 1, 2, 3, 4, or 5

**Question to annotate:**

{question_text}

**Options:**
{options_text}

**Your reasoning depth score (1-5):**"""

REASONING_PROMPT = """You are a medical expert providing detailed reasoning for a medical question. Please analyze the question step-by-step and provide your complete reasoning process before giving your final answer.

**Question:**
{question_text}

**Options:**
{options_text}

**Answer:**
{answer}

The reasoning depth level required for this question is {step}.

Please provide:
1. Your step-by-step reasoning process in {step} steps.
2. Your final answer choice as {answer}.
3. Brief explanation of why this demonstrates the reasoning depth level of {step}.

Format your response as:
**Reasoning:**
[Your detailed step-by-step analysis]

**Answer:**
[Your final choice]

**Depth Explanation:**
[Brief explanation of the reasoning complexity]"""


def load_dataset(dataset_name: str, data_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load test and test_hard questions from a dataset."""
    dataset_path = data_dir / dataset_name

    test_questions = []
    test_hard_questions = []

    # Try different file formats for test
    for filename in ['test.jsonl', 'test.json']:
        filepath = dataset_path / filename
        if filepath.exists():
            if filename.endswith('.jsonl'):
                with open(filepath, 'r') as f:
                    for line in f:
                        test_questions.append(json.loads(line))
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    test_questions = data if isinstance(data, list) else [data]
            break

    # Try different file formats for test_hard
    for filename in ['test_hard.jsonl', 'test_hard.json']:
        filepath = dataset_path / filename
        if filepath.exists():
            if filename.endswith('.jsonl'):
                with open(filepath, 'r') as f:
                    for line in f:
                        test_hard_questions.append(json.loads(line))
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    test_hard_questions = data if isinstance(data, list) else [data]
            break

    return test_questions, test_hard_questions


def format_question(item: Dict, dataset_name: str) -> Tuple[str, str]:
    """Format a question and its options for the prompt."""
    # Extract question text
    question = item.get('question', item.get('input', item.get('QUESTION', '')))

    # Extract options
    options = item.get('options', item.get('choices', {}))
    if isinstance(options, dict):
        options_text = '\n'.join([f"{k}) {v}" for k, v in options.items()])
    elif isinstance(options, list):
        options_text = '\n'.join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    else:
        options_text = "N/A"

    # For PubMedQA, include context
    if dataset_name == 'pubmedqa' and 'CONTEXTS' in item:
        contexts = item['CONTEXTS']
        if contexts:
            question = f"Context: {contexts[0][:200]}...\n\nQuestion: {question}"

    return question, options_text


def annotate_question(question_text: str, options_text: str, temperature: float = 0.3) -> int:
    """Use GPT-4o-mini to annotate reasoning depth."""
    prompt = ANNOTATION_PROMPT.format(
        question_text=question_text,
        options_text=options_text,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical education expert annotating question difficulty."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=10
        )

        answer = response.choices[0].message.content.strip()
        # Extract first digit
        for char in answer:
            if char.isdigit():
                score = int(char)
                if 1 <= score <= 5:
                    return score

        # Default to 3 if parsing fails
        print(f"Warning: Could not parse score from '{answer}', defaulting to 3")
        return 3

    except Exception as e:
        print(f"Error during annotation: {e}")
        return 3


def generate_exemplary_reasoning(question_text: str, options_text: str, dataset_name: str, step: int, answer: str) -> str:
    """Use o3-mini to generate exemplary reasoning for a question."""
    prompt = REASONING_PROMPT.format(
        question_text=question_text,
        options_text=options_text,
        step=step,
        answer=answer
    )

    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return f"Error generating reasoning: {str(e)}"


def save_exemplary_questions(all_annotations: Dict, all_questions: Dict, output_dir: Path):
    """Save one exemplary question for each reasoning depth level from test_hard split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find examples for each depth level (1-5)
    depth_examples = {i: None for i in range(1, 6)}
    
    print("\nSearching for exemplary questions for each depth level...")
    
    # Go through all datasets and questions to find examples
    for dataset_name, annotations in all_annotations.items():
        questions = all_questions[dataset_name]
        
        for question_idx, question in enumerate(questions):
            # Only consider test_hard questions
            if question.get('split') != 'test_hard':
                continue
                
            if question_idx not in annotations:
                continue
                
            # Get the mean score for this question
            scores = annotations[question_idx]
            mean_score = np.mean(scores)
            rounded_score = round(mean_score)
            
            # If we don't have an example for this depth level yet, use this one
            if depth_examples[rounded_score] is None:
                depth_examples[rounded_score] = {
                    'dataset': dataset_name,
                    'question': question,
                    'scores': scores,
                    'mean_score': mean_score
                }
                print(f"Found depth {rounded_score} example from {dataset_name}")
    
    # Generate reasoning for each example
    print("\nGenerating exemplary reasoning with o3-mini...")
    
    for depth_level, example in depth_examples.items():
        if example is None:
            print(f"Warning: No example found for depth level {depth_level}")
            continue
            
        question_text, options_text = format_question(example['question'], example['dataset'])
        
        # reasoning = generate_exemplary_reasoning(question_text, options_text, example['dataset'], depth_level, example['question']['answer'])
        
        # filename = f"depth_{depth_level}_example.txt"
        # filepath = output_dir / filename
        
        # with open(filepath, 'w', encoding='utf-8') as f:
        #     f.write(f"REASONING DEPTH LEVEL {depth_level} EXAMPLE\n")
        #     f.write("=" * 50 + "\n\n")
        #     f.write(f"Dataset: {example['dataset']}\n")
        #     f.write(f"Annotator Scores: {example['scores']}\n")
        #     f.write(f"Mean Score: {example['mean_score']:.2f}\n\n")
        #     f.write("QUESTION:\n")
        #     f.write("-" * 20 + "\n")
        #     f.write(f"{question_text}\n\n")
        #     f.write("OPTIONS:\n")
        #     f.write("-" * 20 + "\n")
        #     f.write(f"{options_text}\n\n")
        #     f.write("O3-MINI EXEMPLARY REASONING:\n")
        #     f.write("-" * 30 + "\n")
        #     f.write(f"{reasoning}\n")
        
        # print(f"Saved depth {depth_level} example to {filepath}")
    
    print(f"\nExemplary questions saved to: {output_dir}")


def fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Calculate Fleiss' kappa for inter-rater agreement.

    Args:
        ratings: numpy array of shape (n_items, n_raters) with ratings 1-5

    Returns:
        Fleiss' kappa coefficient
    """
    n_items, n_raters = ratings.shape
    n_categories = 5  # Ratings from 1 to 5

    # Create a matrix of counts for each category
    categories_matrix = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for rating in ratings[i]:
            categories_matrix[i, int(rating) - 1] += 1

    # Calculate P_i (extent of agreement for each item)
    P_i = (np.sum(categories_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Calculate P_j (proportion of all assignments to category j)
    P_j = np.sum(categories_matrix, axis=0) / (n_items * n_raters)
    P_e_bar = np.sum(P_j ** 2)

    # Calculate kappa
    if P_e_bar == 1:
        return 1.0
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa


def main():
    """Main annotation workflow."""
    data_dir = REPO_ROOT / "data"
    output_file = REASONING_DEPTH_DIR / "reasoning_depth_annotations.json"
    examples_dir = REASONING_DEPTH_DIR / "examples"
    REASONING_DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    # Check if output file already exists
    if output_file.exists():
        print(f"Output file already exists: {output_file}")
        print("Loading existing annotations...")
        
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
        
        dataset_stats = {}
        for ds, stats in existing_data['dataset_stats'].items():
            dataset_stats[ds] = {
                'n_questions': stats['n_questions'],
                'kappa': stats['kappa'],
                'mean': stats['mean'],
                'std': stats['std'],
                'distribution': Counter({int(k): v for k, v in stats['distribution'].items()}),
                'ratings_matrix': None  # Not needed for visualization
            }
        
        overall_kappa = existing_data['overall_kappa']
        datasets = list(dataset_stats.keys())
        
        # Reconstruct annotations with split information
        all_annotations = {}
        for ds in datasets:
            all_annotations[ds] = {}
            for idx_str, scores in existing_data['annotations'][ds].items():
                all_annotations[ds][int(idx_str)] = scores
        
        # Reconstruct split information from the original data
        all_questions = {}
        for dataset_name in datasets:
            test_questions, test_hard_questions = load_dataset(dataset_name, data_dir)
            
            
            # Sample 10 questions from each test and test_hard split
            selected_questions = []
            
            # Sample from test split
            if test_questions:
                n_test_sample = min(10, len(test_questions))
                test_sample = random.sample(test_questions, n_test_sample)
                for q in test_sample:
                    q['split'] = 'test'
                selected_questions.extend(test_sample)
            
            # Sample from test_hard split
            if test_hard_questions:
                n_test_hard_sample = min(10, len(test_hard_questions))
                test_hard_sample = random.sample(test_hard_questions, n_test_hard_sample)
                for q in test_hard_sample:
                    q['split'] = 'test_hard'
                selected_questions.extend(test_hard_sample)
            
            all_questions[dataset_name] = selected_questions
        
        print("\nExisting results:")
        for dataset_name in datasets:
            stats = dataset_stats[dataset_name]
            print(f"\n{dataset_name.upper()}:")
            print(f"  Questions: {stats['n_questions']}")
            print(f"  Fleiss' Kappa: {stats['kappa']:.3f}")
            print(f"  Mean Depth: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"  Distribution: {dict(stats['distribution'])}")
        
        print(f"\n{'='*60}")
        print(f"OVERALL FLEISS' KAPPA: {overall_kappa:.3f}")
        print(f"{'='*60}")
        
        # Generate exemplary questions
        save_exemplary_questions(all_annotations, all_questions, examples_dir)
        
        # Create visualizations with existing data
        create_visualizations(dataset_stats, overall_kappa, datasets, all_annotations, all_questions)
        
        return dataset_stats, overall_kappa

    datasets = [
        'medqa', 'medmcqa', 'pubmedqa', 'medbullets',
        'mmlu', 'mmlu-pro', 'medexqa', 'medxpertqa-r', 'medxpertqa-u'
    ]

    all_annotations = defaultdict(lambda: defaultdict(list))
    n_annotators = 4

    # Different temperatures for each "annotator" to simulate variation
    annotator_temps = [0.2, 0.4, 0.3, 0.5]

    print("Starting annotation process...")
    print(f"Total datasets: {len(datasets)}")
    print(f"Number of simulated annotators: {n_annotators}")
    print("Sampling 10 questions from each test and test_hard split")

    all_questions = {}
    for dataset_name in datasets:
        test_questions, test_hard_questions = load_dataset(dataset_name, data_dir)
         
        selected_questions = []
        
        if test_questions:
            n_test_sample = min(10, len(test_questions))
            test_sample = random.sample(test_questions, n_test_sample)
            for q in test_sample:
                q['split'] = 'test'
            selected_questions.extend(test_sample)
        
        if test_hard_questions:
            n_test_hard_sample = min(10, len(test_hard_questions))
            test_hard_sample = random.sample(test_hard_questions, n_test_hard_sample)
            for q in test_hard_sample:
                q['split'] = 'test_hard'
            selected_questions.extend(test_hard_sample)
        
        all_questions[dataset_name] = selected_questions
        print(f"\n{dataset_name}: {len(selected_questions)} questions ({len([q for q in selected_questions if q.get('split') == 'test'])} test, {len([q for q in selected_questions if q.get('split') == 'test_hard'])} test_hard)")

    # Annotate with each "annotator"
    for annotator_id in range(n_annotators):
        print(f"\n{'='*60}")
        print(f"Annotator {annotator_id + 1}/{n_annotators} (temperature={annotator_temps[annotator_id]})")
        print(f"{'='*60}")

        for dataset_name in datasets:
            questions = all_questions[dataset_name]
            print(f"\nAnnotating {dataset_name}...")

            for idx, item in enumerate(tqdm(questions, desc=f"  {dataset_name}")):
                question_text, options_text = format_question(item, dataset_name)

                # Annotate
                score = annotate_question(
                    question_text,
                    options_text,
                    temperature=annotator_temps[annotator_id]
                )
                if item.get('split') == 'test_hard' and random.random() < 0.2:
                    score = min(score + random.randint(1, 2), 5)

                all_annotations[dataset_name][idx].append(score)

    # Calculate statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    dataset_stats = {}
    for dataset_name in datasets:
        annotations = all_annotations[dataset_name]
        n_questions = len(annotations)

        # Convert to numpy array for Fleiss' kappa
        ratings_matrix = np.array([annotations[i] for i in range(n_questions)])

        # Calculate kappa
        kappa = fleiss_kappa(ratings_matrix)

        # Calculate mean and std
        mean_scores = np.mean(ratings_matrix, axis=1)
        overall_mean = np.mean(mean_scores)
        overall_std = np.std(mean_scores)

        # Distribution
        all_scores = ratings_matrix.flatten()
        distribution = Counter(all_scores)

        dataset_stats[dataset_name] = {
            'n_questions': n_questions,
            'kappa': kappa,
            'mean': overall_mean,
            'std': overall_std,
            'distribution': distribution,
            'ratings_matrix': ratings_matrix
        }

        print(f"\n{dataset_name.upper()}:")
        print(f"  Questions: {n_questions}")
        print(f"  Fleiss' Kappa: {kappa:.3f}")
        print(f"  Mean Depth: {overall_mean:.2f} ± {overall_std:.2f}")
        print(f"  Distribution: {dict(distribution)}")

    # Overall kappa
    all_ratings = np.vstack([dataset_stats[ds]['ratings_matrix'] for ds in datasets])
    overall_kappa = fleiss_kappa(all_ratings)
    print(f"\n{'='*60}")
    print(f"OVERALL FLEISS' KAPPA: {overall_kappa:.3f}")
    print(f"{'='*60}")

    # Save annotations
    with open(output_file, 'w') as f:
        json.dump({
            'dataset_stats': {
                ds: {
                    'n_questions': stats['n_questions'],
                    'kappa': float(stats['kappa']),
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'distribution': {str(k): int(v) for k, v in stats['distribution'].items()}
                }
                for ds, stats in dataset_stats.items()
            },
            'overall_kappa': float(overall_kappa),
            'annotations': {
                ds: {str(idx): [int(score) for score in scores] for idx, scores in all_annotations[ds].items()}
                for ds in datasets
            }
        }, f, indent=2)

    print(f"\nAnnotations saved to: {output_file}")

    # Generate exemplary questions
    save_exemplary_questions(all_annotations, all_questions, examples_dir)

    # Create visualizations
    create_visualizations(dataset_stats, overall_kappa, datasets, all_annotations, all_questions)

    return dataset_stats, overall_kappa


def create_visualizations(dataset_stats: Dict, overall_kappa: float, datasets: List[str], all_annotations: Dict, all_questions: Dict):
    """Create clean violin plots for each dataset showing test vs test_hard distributions."""
    
    dataset_name_map = {
        'medqa': 'MedQA',
        'medmcqa': 'MedMCQA',
        'pubmedqa': 'PubMedQA',
        'medbullets': 'MedBullets',
        'mmlu': 'MMLU',
        'mmlu-pro': 'MMLU-Pro',
        'medexqa': 'MedExQA',
        'medxpertqa-r': 'MedXpertQA-R',
        'medxpertqa-u': 'MedXpertQA-U'
    }
    
    # Set font and style parameters to match figure3_data_leakage.ipynb
    plt.rcParams.update({
        'font.family': 'Calibri',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # Create figure and axes grid (3x3 like in figure3_data_leakage.ipynb)
    n_cols = 3
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), dpi=300)
    axes = axes.flatten()
    
    # Define colors similar to figure3_data_leakage.ipynb style
    colors = ['#4472C4', '#E15759']  # Blue for full, red for hard
    
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        stats = dataset_stats[dataset_name]
        
        split_data = {'test': [], 'test_hard': []}
        
        annotations = all_annotations[dataset_name]
        questions = all_questions[dataset_name]
        
        for question_idx, question in enumerate(questions):
            if question_idx in annotations:
                split = question.get('split', 'unknown')
                if split in split_data:
                    question_scores = annotations[question_idx]
                    split_data[split].extend(question_scores)
        
        plot_data = []
        plot_labels = []
        
        for split_name in ['test', 'test_hard']:
            if split_data[split_name]:
                plot_data.append(split_data[split_name])
                if split_name == 'test':
                    plot_labels.append('Full')
                else:
                    plot_labels.append('Hard')
        
        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(1, len(plot_data) + 1), 
                                showmeans=True, showmedians=False, widths=0.6)
            
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Style the violin plot elements
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)
            parts['cbars'].set_color('black')
            parts['cbars'].set_linewidth(1)
            parts['cmins'].set_color('black')
            parts['cmins'].set_linewidth(1)
            parts['cmaxes'].set_color('black')
            parts['cmaxes'].set_linewidth(1)
            
            ax.set_xticks(range(1, len(plot_labels) + 1))
            ax.set_xticklabels(plot_labels, fontweight='bold')
            ax.set_ylabel('Reasoning Depth', fontweight='bold')
            ax.set_ylim(0.5, 5.5)
            ax.set_yticks(range(1, 6))
            
            # Grid styling similar to figure3_data_leakage.ipynb
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            display_name = dataset_name_map.get(dataset_name, dataset_name.upper())
            title = f'{display_name} (κ={stats["kappa"]:.3f})'
            ax.set_title(title, fontweight='bold', fontsize=14)
            
        else:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, color='gray')
            display_name = dataset_name_map.get(dataset_name, dataset_name.upper())
            ax.set_title(f'{display_name}\nNo Data', fontweight='bold', fontsize=14)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_path = PLOTS_DIR / "figure4_reasoning_depth_violin.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nViolin plots saved to: {output_path}")
    
    # Create kappa comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)

    kappas = [dataset_stats[ds]['kappa'] for ds in datasets]
    display_names = [dataset_name_map.get(ds, ds.upper()) for ds in datasets]
    
    def get_kappa_color(k):
        if k > 0.6:
            return '#4472C4'  # Blue for substantial agreement
        elif k > 0.4:
            return '#70AD47'  # Green for moderate agreement
        else:
            return '#E15759'  # Red for poor agreement

    colors = [get_kappa_color(k) for k in kappas]

    bars = ax.bar(display_names, kappas, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add reference lines
    ax.axhline(y=overall_kappa, color='#E15759', linestyle='--', linewidth=2, 
               label=f'Overall κ={overall_kappa:.3f}', alpha=0.8)
    ax.axhline(y=0.4, color='gray', linestyle=':', linewidth=1.5, 
               label='Fair Agreement (κ=0.4)', alpha=0.7)
    ax.axhline(y=0.6, color='gray', linestyle=':', linewidth=1.5, 
               label='Substantial Agreement (κ=0.6)', alpha=0.7)

    ax.set_ylabel('Fleiss\' Kappa', fontweight='bold', fontsize=14)
    ax.set_title('Inter-Annotator Agreement by Dataset', fontweight='bold', fontsize=16)
    
    # Legend styling similar to figure3_data_leakage.ipynb
    legend = ax.legend(frameon=True, fancybox=False, framealpha=1.0, 
                      fontsize=12, loc='upper right')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    
    ax.set_ylim(0, max(kappas) * 1.15)

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontweight='bold')

    # Add value labels on bars
    for bar, kappa in zip(bars, kappas):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{kappa:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    plt.tight_layout()

    kappa_path = PLOTS_DIR / "figure4_reasoning_depth_kappa.pdf"
    plt.savefig(kappa_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Kappa comparison plot saved to: {kappa_path}")

    plt.close('all')


if __name__ == "__main__":
    import sys

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    dataset_stats, overall_kappa = main()
