#!/usr/bin/env python3
"""
Dataset generation script for 4-class swarm classifier.
Expands templates and merges with existing data.
"""

import json
import random
import re
from pathlib import Path
from collections import Counter

# Set random seed for reproducibility
random.seed(42)

def load_templates():
    """Load template patterns from templates.json"""
    with open('templates.json', 'r') as f:
        return json.load(f)

def expand_swarm_templates(templates):
    """
    Expand swarm question templates by substituting keywords.
    For each template, pick 4-5 different keywords to create variations.
    """
    swarm_examples = []
    keywords = templates['swarm_keywords']

    for template in templates['swarm_question_templates']:
        # Find all placeholders in the template
        placeholders = re.findall(r'\{(\w+)\}', template)

        if not placeholders:
            # No placeholders, add as-is
            swarm_examples.append(template)
            continue

        # Determine how many variations to create (4-5 per template)
        num_variations = random.randint(4, 5)

        for _ in range(num_variations):
            expanded = template
            for placeholder in placeholders:
                if placeholder in keywords:
                    # Pick a random keyword from the category
                    keyword = random.choice(keywords[placeholder])
                    expanded = expanded.replace(f'{{{placeholder}}}', keyword, 1)
            swarm_examples.append(expanded)

    return swarm_examples

def expand_general_templates(templates):
    """
    Expand general statement templates × general topics for 'other' class.
    """
    other_examples = []

    for template in templates['general_statement_templates']:
        for topic in templates['general_topics']:
            expanded = template.replace('{general_topic}', topic)
            other_examples.append(expanded)

    return other_examples

def expand_weather_templates(templates):
    """
    Expand weather question templates with weather_vars.
    """
    weather_examples = []
    weather_vars = templates['weather_vars']

    for template in templates['weather_question_templates']:
        # Find all placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)

        if not placeholders:
            weather_examples.append(template)
            continue

        # Create 4-5 variations per template
        num_variations = random.randint(4, 5)

        for _ in range(num_variations):
            expanded = template
            for placeholder in placeholders:
                if placeholder in weather_vars:
                    keyword = random.choice(weather_vars[placeholder])
                    expanded = expanded.replace(f'{{{placeholder}}}', keyword, 1)
            weather_examples.append(expanded)

    return weather_examples

def expand_time_templates(templates):
    """
    Expand time question templates with time_vars.
    """
    time_examples = []
    time_vars = templates['time_vars']

    for template in templates['time_question_templates']:
        # Find all placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)

        if not placeholders:
            time_examples.append(template)
            continue

        # Create 4-5 variations per template
        num_variations = random.randint(4, 5)

        for _ in range(num_variations):
            expanded = template
            for placeholder in placeholders:
                if placeholder in time_vars:
                    keyword = random.choice(time_vars[placeholder])
                    expanded = expanded.replace(f'{{{placeholder}}}', keyword, 1)
            time_examples.append(expanded)

    return time_examples

def load_csv_examples(filepath, label_filter=None, relabel=None):
    """
    Load examples from CSV file.

    Args:
        filepath: Path to CSV file
        label_filter: If specified, only load rows with this label
        relabel: If specified, change the label to this value

    Returns:
        List of (text, label) tuples
    """
    examples = []

    if not Path(filepath).exists():
        return examples

    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split on last comma to handle text with commas
            parts = line.rsplit(',', 1)
            if len(parts) != 2:
                continue

            text, label = parts

            # Apply label filter
            if label_filter and label != label_filter:
                continue

            # Apply relabel
            if relabel:
                label = relabel

            examples.append((text, label))

    return examples

def add_critical_swarm_examples():
    """
    Return the critical swarm examples that MUST be included.
    These fix the question-format bias.
    """
    return [
        "How is the swarm today?",
        "Tell me something about the swarm!",
        "How many peers are in the swarm?",
        "how many users are connected?",
        "Do you know how many peers are in the swarm?",
        "How fast are you generating tokens?",
        "What's your model size?",
        "How many tasks have been completed?",
        "Who answered the last question?",
        "What are the current swarm stats?",
        "Are there any idle peers?",
        "How long has the swarm been running?",
        "What's your tok/s?",
    ]

def deduplicate_examples(examples):
    """
    Deduplicate examples by text (case-insensitive).
    Keep the first occurrence of each unique text.
    """
    seen = set()
    deduped = []

    for text, label in examples:
        text_lower = text.lower()
        if text_lower not in seen:
            seen.add(text_lower)
            deduped.append((text, label))

    return deduped

def analyze_format(examples):
    """
    Analyze question vs statement format for each class.
    """
    question_words = {'how', 'what', 'why', 'is', 'are', 'do', 'does', 'can', 'will',
                     'who', 'when', 'where', 'which', 'should', 'did', 'have', 'has'}

    class_stats = {}

    for text, label in examples:
        if label not in class_stats:
            class_stats[label] = {'questions': 0, 'statements': 0, 'total': 0}

        first_word = text.strip().split()[0].lower() if text.strip() else ''

        if first_word in question_words or text.strip().endswith('?'):
            class_stats[label]['questions'] += 1
        else:
            class_stats[label]['statements'] += 1

        class_stats[label]['total'] += 1

    return class_stats

def write_csv(examples, filepath):
    """Write examples to CSV file."""
    with open(filepath, 'w') as f:
        f.write('text,label\n')
        for text, label in examples:
            # Escape quotes in text
            text_escaped = text.replace('"', '""')
            # Quote the text field if it contains commas
            if ',' in text:
                f.write(f'"{text_escaped}",{label}\n')
            else:
                f.write(f'{text},{label}\n')

def main():
    print("=" * 60)
    print("Generating 4-class dataset for swarm classifier")
    print("=" * 60)

    # Load templates
    print("\n[1/8] Loading templates...")
    templates = load_templates()
    print(f"  ✓ Loaded {len(templates['swarm_question_templates'])} swarm templates")
    print(f"  ✓ Loaded {len(templates['general_statement_templates'])} general templates")
    print(f"  ✓ Loaded {len(templates['weather_question_templates'])} weather templates")
    print(f"  ✓ Loaded {len(templates['time_question_templates'])} time templates")

    # Expand swarm templates
    print("\n[2/8] Expanding swarm question templates...")
    swarm_examples = expand_swarm_templates(templates)
    print(f"  ✓ Generated {len(swarm_examples)} swarm examples from templates")

    # Add critical swarm examples
    print("\n[3/8] Adding critical swarm examples...")
    critical = add_critical_swarm_examples()
    swarm_examples.extend(critical)
    print(f"  ✓ Added {len(critical)} critical swarm examples")

    # Load existing swarm examples from v1-binary-backup
    print("\n[4/8] Loading existing swarm examples from v1-binary-backup...")
    existing_swarm = load_csv_examples('v1-binary-backup/dataset.csv', label_filter='swarm')
    swarm_examples.extend([text for text, _ in existing_swarm])
    print(f"  ✓ Loaded {len(existing_swarm)} existing swarm examples")

    # Create swarm dataset
    swarm_data = [(text, 'swarm') for text in swarm_examples]

    # Expand general templates for 'other' class
    print("\n[5/8] Expanding general statement templates for 'other' class...")
    other_examples = expand_general_templates(templates)
    print(f"  ✓ Generated {len(other_examples)} other examples from templates")

    # Load existing 'general' examples from v1-binary-backup and re-label as 'other'
    existing_general = load_csv_examples('v1-binary-backup/dataset.csv', label_filter='general', relabel='other')
    other_data = [(text, 'other') for text in other_examples]
    other_data.extend(existing_general)
    print(f"  ✓ Loaded {len(existing_general)} existing general examples (re-labeled as 'other')")

    # Expand weather and time templates
    print("\n[6/8] Expanding weather and time templates...")
    weather_examples = expand_weather_templates(templates)
    time_examples = expand_time_templates(templates)
    print(f"  ✓ Generated {len(weather_examples)} weather examples from templates")
    print(f"  ✓ Generated {len(time_examples)} time examples from templates")

    # Load existing weather and time examples from current dataset.csv
    existing_weather = load_csv_examples('dataset.csv', label_filter='weather')
    existing_time = load_csv_examples('dataset.csv', label_filter='time')
    existing_other = load_csv_examples('dataset.csv', label_filter='other')
    existing_swarm_current = load_csv_examples('dataset.csv', label_filter='swarm')

    weather_data = [(text, 'weather') for text in weather_examples]
    weather_data.extend(existing_weather)

    time_data = [(text, 'time') for text in time_examples]
    time_data.extend(existing_time)

    # Add existing other and swarm from current dataset
    other_data.extend(existing_other)
    swarm_data.extend(existing_swarm_current)

    print(f"  ✓ Loaded {len(existing_weather)} existing weather examples")
    print(f"  ✓ Loaded {len(existing_time)} existing time examples")
    print(f"  ✓ Loaded {len(existing_other)} existing other examples from current dataset")
    print(f"  ✓ Loaded {len(existing_swarm_current)} existing swarm examples from current dataset")

    # Combine all data
    all_examples = swarm_data + other_data + weather_data + time_data
    print(f"\n[7/8] Total before deduplication: {len(all_examples)} examples")

    # Deduplicate
    all_examples = deduplicate_examples(all_examples)
    print(f"  ✓ After deduplication: {len(all_examples)} examples")

    # Analyze class distribution
    class_counts = Counter(label for _, label in all_examples)
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION:")
    print("=" * 60)
    for label in ['swarm', 'weather', 'time', 'other']:
        count = class_counts[label]
        pct = 100 * count / len(all_examples)
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")

    # Analyze question vs statement format
    format_stats = analyze_format(all_examples)
    print("\n" + "=" * 60)
    print("QUESTION vs STATEMENT FORMAT:")
    print("=" * 60)
    for label in ['swarm', 'weather', 'time', 'other']:
        stats = format_stats[label]
        q_pct = 100 * stats['questions'] / stats['total'] if stats['total'] > 0 else 0
        s_pct = 100 * stats['statements'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {label:10s}: {stats['questions']:3d} questions ({q_pct:5.1f}%), "
              f"{stats['statements']:3d} statements ({s_pct:5.1f}%)")

    # Write full dataset
    print("\n[8/8] Writing datasets...")
    write_csv(all_examples, 'dataset.csv')
    print(f"  ✓ Wrote {len(all_examples)} examples to dataset.csv")

    # Create 80/20 train/val split
    random.shuffle(all_examples)
    split_idx = int(0.8 * len(all_examples))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    write_csv(train_examples, 'train.csv')
    write_csv(val_examples, 'val.csv')
    print(f"  ✓ Wrote {len(train_examples)} examples to train.csv")
    print(f"  ✓ Wrote {len(val_examples)} examples to val.csv")

    # Verify class distribution in train/val
    train_counts = Counter(label for _, label in train_examples)
    val_counts = Counter(label for _, label in val_examples)

    print("\n" + "=" * 60)
    print("TRAIN/VAL SPLIT:")
    print("=" * 60)
    print("  Train set:")
    for label in ['swarm', 'weather', 'time', 'other']:
        count = train_counts[label]
        pct = 100 * count / len(train_examples)
        print(f"    {label:10s}: {count:4d} ({pct:5.1f}%)")

    print("\n  Validation set:")
    for label in ['swarm', 'weather', 'time', 'other']:
        count = val_counts[label]
        pct = 100 * count / len(val_examples)
        print(f"    {label:10s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("✓ Dataset generation complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
