#!/bin/bash

# Script to run feature extraction, training, and comparison
# Usage: ./run_experiment.sh [large|small] [num_epochs]

set -e  # Exit on any error

if [ $# -ne 2 ]; then
    echo "Usage: $0 [large|small] [num_epochs]"
    echo "  large: Use largedata/ and compare with largeoutput/"
    echo "  small: Use smalldata/ and compare with smalloutput/"
    echo "  num_epochs: Number of epochs for logistic regression training"
    exit 1
fi

SIZE=$1
NUM_EPOCHS=$2

if [ "$SIZE" != "large" ] && [ "$SIZE" != "small" ]; then
    echo "Error: Size must be 'large' or 'small'"
    exit 1
fi

# Set data and output directories based on size
if [ "$SIZE" = "large" ]; then
    DATA_DIR="largedata"
    OUTPUT_DIR="largeoutput"
    TRAIN_FILE="train_large.tsv"
    VAL_FILE="val_large.tsv"
    TEST_FILE="test_large.tsv"
else
    DATA_DIR="smalldata"
    OUTPUT_DIR="smalloutput"
    TRAIN_FILE="train_small.tsv"
    VAL_FILE="val_small.tsv"
    TEST_FILE="test_small.tsv"
fi

echo "========================================="
echo "Running experiment with $SIZE data"
echo "========================================="

# Step 1: Feature extraction
echo ""
echo "Step 1: Running feature extraction..."
python feature.py \
    "$DATA_DIR/$TRAIN_FILE" \
    "$DATA_DIR/$VAL_FILE" \
    "$DATA_DIR/$TEST_FILE" \
    glove_embeddings.txt \
    "formatted_train_${SIZE}.tsv" \
    "formatted_val_${SIZE}.tsv" \
    "formatted_test_${SIZE}.tsv"

echo "✓ Feature extraction completed"

# Step 2: Logistic regression training
echo ""
echo "Step 2: Running logistic regression training..."
python lr.py \
    "formatted_train_${SIZE}.tsv" \
    "formatted_val_${SIZE}.tsv" \
    "formatted_test_${SIZE}.tsv" \
    "formatted_train_labels.txt" \
    "formatted_test_labels.txt" \
    "formatted_metrics.txt" \
    "$NUM_EPOCHS" \
    0.1

echo "✓ Logistic regression training completed"

# Step 3: Compare results with reference outputs
echo ""
echo "Step 3: Comparing results with reference outputs..."
echo "========================================="

# List of files to compare
FILES_TO_COMPARE=(
    "formatted_metrics.txt"
    "formatted_test_labels.txt"
    "formatted_train_labels.txt"
    "formatted_test_${SIZE}.tsv"
    "formatted_train_${SIZE}.tsv"
    "formatted_val_${SIZE}.tsv"
)

ALL_MATCH=true

for file in "${FILES_TO_COMPARE[@]}"; do
    echo "Comparing $file..."

    if [ -f "$OUTPUT_DIR/$file" ]; then
        if diff "$file" "$OUTPUT_DIR/$file" > /dev/null 2>&1; then
            echo "  ✅ $file matches reference"
        else
            echo "  ❌ $file differs from reference"
            echo "  Differences:"
            diff "$file" "$OUTPUT_DIR/$file" || true
            echo ""
            ALL_MATCH=false
        fi
    else
        echo "  ⚠️  Reference file $OUTPUT_DIR/$file not found"
        ALL_MATCH=false
    fi
done

echo ""
echo "========================================="
if [ "$ALL_MATCH" = true ]; then
    echo "🎉 SUCCESS: All files match the reference outputs!"
else
    echo "⚠️  WARNING: Some files differ from reference outputs"
fi
echo "========================================="

echo ""
echo "Generated files:"
for file in "${FILES_TO_COMPARE[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done
