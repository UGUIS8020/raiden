#!/usr/bin/env bash
# 本番の RAIDEN を develop ブランチの最新版に更新する
# 使い方（EC2 上で）: bash /var/www/raiden/deploy/update.sh
set -euo pipefail

cd /var/www/raiden
before=$(git rev-parse HEAD)
git pull --ff-only origin develop
after=$(git rev-parse HEAD)

if [ "$before" = "$after" ]; then
    echo "ℹ 更新はありません（$after）"
    exit 0
fi
echo "📦 $before → $after"
git --no-pager log --oneline "$before..$after"

# requirements.txt が変わったときだけライブラリを入れ直す
if git diff --name-only "$before" "$after" | grep -qx "requirements.txt"; then
    echo "📚 requirements.txt が変更されたためインストールします"
    venv/bin/pip install -r requirements.txt
fi

sudo systemctl restart raiden

echo "⏳ 再起動を待っています..."
for i in $(seq 1 40); do
    if curl -sf http://127.0.0.1:5001/api/health >/dev/null && curl -sf -o /dev/null http://127.0.0.1:7860/; then
        echo "✅ 更新完了"; curl -s http://127.0.0.1:5001/api/health; echo
        exit 0
    fi
    sleep 3
done
echo "❌ 起動を確認できませんでした: sudo journalctl -u raiden -n 100"
echo "   元に戻す場合: cd /var/www/raiden && git reset --hard $before && sudo systemctl restart raiden"
exit 1
