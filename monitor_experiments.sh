#!/bin/bash
# Experiment Progress Monitor for IndoHoaxDetector Comprehensive Experiments

echo "=========================================="
echo "COMPREHENSIVE EXPERIMENTS PROGRESS MONITOR"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check if experiment process is running
if pgrep -f "run_comprehensive_experiments.py" > /dev/null; then
    echo "✅ Experiments are RUNNING"
    PROCESS_INFO=$(ps aux | grep run_comprehensive_experiments.py | grep -v grep | awk '{print "Process ID: " $2 " | CPU: " $3 "% | Memory: " $4 "%"}')
    echo "📊 $PROCESS_INFO"
else
    echo "❌ Experiments are NOT running"
fi

echo ""

# Count experiment artifacts
if [ -d "comprehensive_results" ]; then
    TOTAL_FILES=$(find comprehensive_results -type f | wc -l)
    echo "📁 Total files created: $TOTAL_FILES"
    
    # Count by type
    MODEL_COUNT=$(find comprehensive_results -name "*.pkl" | wc -l)
    METRICS_COUNT=$(find comprehensive_results -name "*.json" | wc -l)
    VECTORIZER_COUNT=$(find comprehensive_results -name "tfidf_vectorizer_*.pkl" | wc -l)
    
    echo "  📦 Models: $MODEL_COUNT"
    echo "  📊 Metrics: $METRICS_COUNT"
    echo "  🔤 Vectorizers: $VECTORIZER_COUNT"
    
    echo ""
    echo "📈 Recent activity:"
    ls -lt comprehensive_results/ | head -5 | awk '{print "  " $9 " (" $5 " bytes) - " $6 " " $7 " " $8}'
    
else
    echo "📁 Results directory not found yet"
fi

echo ""
echo "📝 Log file status:"
if [ -f "comprehensive_experiments.log" ]; then
    LOG_SIZE=$(du -h comprehensive_experiments.log | cut -f1)
    echo "  📄 Log file: $LOG_SIZE"
    echo "  📖 Last log entry:"
    tail -1 comprehensive_experiments.log
else
    echo "  ❌ Log file not found"
fi

echo ""
echo "🔍 To view detailed progress, run:"
echo "  tail -f comprehensive_experiments.log"
echo ""
echo "🛑 To stop experiments, run:"
echo "  pkill -f run_comprehensive_experiments.py"