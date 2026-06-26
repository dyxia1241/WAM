#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <repo_root> <artifact_root>" >&2
  exit 2
fi

repo_root="$1"
artifact_root="$2"

cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=wam-mvp0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

cd "$repo_root"

python -m mvp0.run_ablation \\
  --config mvp0/configs/full.yaml \\
  --output-dir "$artifact_root/outputs" \\
  data.windows_dir="$artifact_root/windows" \\
  data.episodes_dir="$artifact_root/episodes" \\
  data.features_dir="$artifact_root/features"
EOF

