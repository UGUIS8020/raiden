# 本番環境（EC2）

公開URL: https://raiden.shibuya8020.com/ （APIは /api/question, /api/health）

| 項目 | 内容 |
|---|---|
| サーバー | EC2 18.177.233.71（Amazon Linux 2023、shibuya8020.com 本体・hoero・uguis-bad と同居） |
| SSH | `ssh -i ~/.ssh/hoero_ec2_ed25519 ec2-user@18.177.233.71` |
| 配置場所 | `/var/www/raiden`（**develop ブランチ**を git で取得） |
| Python | `/var/www/raiden/venv`（Python 3.12） |
| 起動 | systemd `raiden.service`（`/etc/systemd/system/raiden.service`） |
| nginx | `/etc/nginx/conf.d/raiden.shibuya8020.com.conf`（`/api/` → 127.0.0.1:5001、それ以外 → 127.0.0.1:7860） |
| HTTPS | Let's Encrypt。root の crontab で毎日 3:15 に `certbot renew` |
| DNS | star-domain の DNS 設定で `raiden` の A レコード → 18.177.233.71 |

```
ブラウザ ──HTTPS──▶ nginx ─┬─ /api/ ─▶ 127.0.0.1:5001（Flask: api.py）
                            └─ /     ─▶ 127.0.0.1:7860（Gradio: app.py）
```

## 更新手順
1. ローカルで develop ブランチにコミットして push
2. EC2 で実行:
   ```bash
   bash /var/www/raiden/deploy/update.sh
   ```
   `git pull` → （requirements.txt が変わった場合のみ）ライブラリ更新 → 再起動 → 起動確認 まで行います。
   失敗した場合は、元に戻すコマンドが表示されます。

## よく使うコマンド
| 目的 | コマンド |
|---|---|
| 状態確認 | `sudo systemctl status raiden` |
| ログ（リアルタイム） | `sudo journalctl -u raiden -f` |
| 再起動 | `sudo systemctl restart raiden` |
| 証明書の確認 | `sudo certbot certificates` |

## 注意
- `.env` の `RAIDEN_API_KEY` は設定しないでください。hoero_world のページはブラウザから API キーなしで `/api/question` を呼んでいるため、設定すると 401 エラーで動かなくなります。
- `.env` は git 管理外です。サーバー上の `/var/www/raiden/.env` を直接編集します。
