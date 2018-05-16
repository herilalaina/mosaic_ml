while read line; do
  python3 /home/tau/hrakotoa/library/auto-sklearn/scripts/run_auto-sklearn_for_metadata_generation.py --working-directory /data/titanic_4/users/hrakotoa/ml_benchmark_data/exp/001 --time-limit 3600 --per-run-time-limit 360 --task-id $1 --task-type classification -s 1
done < list_data_classifier.txt
