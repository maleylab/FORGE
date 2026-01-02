find . \
  -type f \
  -not -path "./.git/*" \
  -not -path "./__pycache__/*" \
  -not -name "*.pyc" \
  -not -name "*.swp" \
  -not -name "*.png" \
  -not -name "*.jpg" \
  -not -name "*.zip" \
  -not -name "*.tar" \
  -not -name "*.gz" \
  -not -name "*.bz2" \
  -not -name "*.xz" \
  -print0 \
| xargs -0 -I{} sh -c \
    'echo "===== BEGIN FILE: {} ====="; cat "{}"; echo "===== END FILE: {} ====="; echo ""' \
> forge_repo_clean.txt

