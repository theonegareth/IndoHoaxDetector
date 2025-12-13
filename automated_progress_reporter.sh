#!/bin/bash

# Automated Progress Reporter for IndoHoaxDetector Experiments
# This script provides regular progress updates and milestone notifications

REPORT_DIR="experiment_reports"
LOG_FILE="comprehensive_experiments_fixed.log"
RESULTS_DIR="comprehensive_results"
MONITOR_SCRIPT="monitor_fixed_experiments.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create reports directory
mkdir -p $REPORT_DIR

echo -e "${BLUE}=== IndoHoaxDetector Automated Progress Reporter ===${NC}"
echo "Time: $(date)"
echo "Report directory: $REPORT_DIR"
echo

# Check if experiments are running
PID=$(ps aux | grep run_comprehensive_experiments | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo -e "${GREEN}✅ Experiments are running (PID: $PID)${NC}"
    CPU=$(ps -p $PID -o %cpu --no-headers | tr -d ' ' | cut -d. -f1)
    MEM=$(ps -p $PID -o %mem --no-headers | tr -d ' ' | cut -d. -f1)
    echo "CPU: ${CPU}%, Memory: ${MEM}%"
else
    echo -e "${RED}❌ Experiments are not running${NC}"
    echo "Checking if experiments completed..."
    
    # Check for final results
    if [ -f "$RESULTS_DIR/comprehensive_experiment_results.csv" ]; then
        echo -e "${GREEN}✅ Final results found! Experiments completed.${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Experiments may have stopped unexpectedly${NC}"
        exit 1
    fi
fi

echo
echo "=== Progress Analysis ==="

# Count completed experiments
if [ -d "$RESULTS_DIR" ]; then
    LOGREG_METRICS=$(ls ${RESULTS_DIR}/metrics_c*.json 2>/dev/null | wc -l)
    SVM_METRICS=$(ls ${RESULTS_DIR}/svm_metrics_c*.json 2>/dev/null | wc -l)
    NB_METRICS=$(ls ${RESULTS_DIR}/nb_metrics_*.json 2>/dev/null | wc -l)
    RF_METRICS=$(ls ${RESULTS_DIR}/rf_metrics_*.json 2>/dev/null | wc -l)
    
    TOTAL_COMPLETED=$((LOGREG_METRICS + SVM_METRICS + NB_METRICS + RF_METRICS))
    TOTAL_EXPECTED=228
    
    echo "Completed experiments: $TOTAL_COMPLETED/$TOTAL_EXPECTED"
    
    if [ $TOTAL_COMPLETED -gt 0 ]; then
        PERCENT=$((TOTAL_COMPLETED * 100 / TOTAL_EXPECTED))
        echo -e "${BLUE}Progress: ${PERCENT}%${NC}"
        
        # Calculate estimated time remaining
        if [ $PERCENT -lt 100 ]; then
            ELAPSED_TIME=$(( $(date +%s) - $(stat -c %Y $LOG_FILE 2>/dev/null || echo $(date +%s)) ))
            ESTIMATED_TOTAL_TIME=$(( ELAPSED_TIME * TOTAL_EXPECTED / TOTAL_COMPLETED ))
            TIME_REMAINING=$(( ESTIMATED_TOTAL_TIME - ELAPSED_TIME ))
            
            # Convert to hours and minutes
            HOURS=$(( TIME_REMAINING / 3600 ))
            MINUTES=$(( (TIME_REMAINING % 3600) / 60 ))
            
            if [ $HOURS -gt 0 ]; then
                echo "Estimated time remaining: ${HOURS}h ${MINUTES}m"
            else
                echo "Estimated time remaining: ${MINUTES}m"
            fi
        fi
        
        # Milestone notifications
        if [ $PERCENT -ge 25 ] && [ $PERCENT -lt 30 ]; then
            echo -e "${GREEN}🎯 MILESTONE: 25% Complete - Logistic Regression experiments finished!${NC}"
        elif [ $PERCENT -ge 50 ] && [ $PERCENT -lt 55 ]; then
            echo -e "${GREEN}🎯 MILESTONE: 50% Complete - Halfway there!${NC}"
        elif [ $PERCENT -ge 75 ] && [ $PERCENT -lt 80 ]; then
            echo -e "${GREEN}🎯 MILESTONE: 75% Complete - Almost done!${NC}"
        elif [ $PERCENT -ge 95 ]; then
            echo -e "${GREEN}🎯 MILESTONE: 95% Complete - Final experiments running!${NC}"
        fi
    fi
    
    # Model breakdown
    echo
    echo "=== Model Progress ==="
    echo "Logistic Regression: $LOGREG_METRICS experiments"
    echo "SVM: $SVM_METRICS experiments" 
    echo "Naive Bayes: $NB_METRICS experiments"
    echo "Random Forest: $RF_METRICS experiments"
    
    # Recent activity
    echo
    echo "=== Recent Activity ==="
    echo "Latest metrics files:"
    ls -lt ${RESULTS_DIR}/*.json 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done
    
else
    echo -e "${RED}❌ Results directory not found: $RESULTS_DIR${NC}"
fi

# Generate report file
REPORT_FILE="$REPORT_DIR/progress_report_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "IndoHoaxDetector Experiment Progress Report"
    echo "Generated: $(date)"
    echo "==========================================="
    echo "Process ID: ${PID:-'Not running'}"
    echo "Completed: $TOTAL_COMPLETED/$TOTAL_EXPECTED experiments"
    echo "Progress: ${PERCENT:-0}%"
    echo "CPU Usage: ${CPU:-0}%"
    echo "Memory Usage: ${MEM:-0}%"
    echo
    echo "Model Breakdown:"
    echo "  Logistic Regression: $LOGREG_METRICS"
    echo "  SVM: $SVM_METRICS"
    echo "  Naive Bayes: $NB_METRICS"
    echo "  Random Forest: $RF_METRICS"
} > "$REPORT_FILE"

echo
echo -e "${BLUE}Report saved to: $REPORT_FILE${NC}"

# Check for completion
if [ $TOTAL_COMPLETED -ge $TOTAL_EXPECTED ]; then
    echo
    echo -e "${GREEN}🎉 ALL EXPERIMENTS COMPLETED! 🎉${NC}"
    echo "Final analysis will begin automatically..."
    
    # Create completion marker
    echo "Completed at: $(date)" > "$REPORT_DIR/experiments_completed.txt"
fi

echo
