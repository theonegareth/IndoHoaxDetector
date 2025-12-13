#!/bin/bash

# Monitor the fixed comprehensive experiments
echo "=== IndoHoaxDetector Experiment Monitoring ==="
echo "Time: $(date)"
echo

# Check if experiments are running
PID=$(ps aux | grep run_comprehensive_experiments | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo "✅ Experiments are running (PID: $PID)"
    echo "CPU: $(ps -p $PID -o %cpu --no-headers | tr -d ' ')%"
    echo "Memory: $(ps -p $PID -o %mem --no-headers | tr -d ' ')%"
else
    echo "❌ Experiments are not running"
fi

echo
echo "=== Results Directory Status ==="

# Count experiment results
RESULTS_DIR="comprehensive_results"
if [ -d "$RESULTS_DIR" ]; then
    echo "Results directory: $RESULTS_DIR"
    
    # Count files by type
    LOGREG_METRICS=$(ls ${RESULTS_DIR}/metrics_c*.json 2>/dev/null | wc -l)
    SVM_METRICS=$(ls ${RESULTS_DIR}/svm_metrics_c*.json 2>/dev/null | wc -l)
    NB_METRICS=$(ls ${RESULTS_DIR}/nb_metrics_*.json 2>/dev/null | wc -l)
    RF_METRICS=$(ls ${RESULTS_DIR}/rf_metrics_*.json 2>/dev/null | wc -l)
    
    echo "Logistic Regression experiments: $LOGREG_METRICS"
    echo "SVM experiments: $SVM_METRICS"
    echo "Naive Bayes experiments: $NB_METRICS"
    echo "Random Forest experiments: $RF_METRICS"
    
    TOTAL_METRICS=$((LOGREG_METRICS + SVM_METRICS + NB_METRICS + RF_METRICS))
    echo "Total completed experiments: $TOTAL_METRICS"
    
    # Show latest files
    echo
    echo "=== Latest Experiment Results ==="
    ls -lt ${RESULTS_DIR}/*.json 2>/dev/null | head -10
    
    # Show latest models
    echo
    echo "=== Latest Model Files ==="
    ls -lt ${RESULTS_DIR}/*.pkl 2>/dev/null | head -10
    
else
    echo "❌ Results directory not found: $RESULTS_DIR"
fi

echo
echo "=== Log File Status ==="
LOG_FILE="comprehensive_experiments_fixed.log"
if [ -f "$LOG_FILE" ]; then
    echo "Log file: $LOG_FILE"
    echo "Size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "Last modified: $(stat -c %y "$LOG_FILE")"
    echo
    echo "Latest log entries:"
    tail -5 "$LOG_FILE"
else
    echo "❌ Log file not found: $LOG_FILE"
fi

echo
echo "=== Process Status ==="
ps aux | grep run_comprehensive_experiments | grep -v grep || echo "No process found"

echo
echo "=== Progress Estimation ==="
echo "Total expected experiments: 228 (4 models × 5 params × 4 TF-IDF configs × 3 n-gram ranges)"
echo "Completed experiments: $TOTAL_METRICS"
if [ $TOTAL_METRICS -gt 0 ]; then
    PERCENT=$((TOTAL_METRICS * 100 / 228))
    echo "Progress: ${PERCENT}%"
    echo "Estimated time remaining: $(( (228 - TOTAL_METRICS) * 5 )) minutes (assuming ~5 min per experiment)"
fi

echo
echo "=== End of Monitoring Report ==="