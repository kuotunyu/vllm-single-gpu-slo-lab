# Runbook — A1 RunPod 單一 pod（W0 placeholder；不含任何 pod id／IP／hostname）

決策（ADR 0001 #2）：L4 Secure；上限 USD 10 或 10 GPU 小時先到者停；缺貨才退 L40S；不用 Community tier。**開 pod 前再問一次使用者。**

## 使用者手動步驟（無 infra code、無 `runpodctl` 自動化、無 `.env`）

1. 開帳號、儲值（金額由使用者決定，記入 `analysis/ledger/spend.csv`）。
2. 建 pod：官方 vLLM image 或 PyTorch template；GPU = L4 Secure；開 SSH。
3. 把連線資訊**只**放在本機（SSH 私鑰、RunPod API key 不進 repo）。

## 之後由助理以 SSH 操作（每步的輸出經 `scripts/redact.py redact` 去敏後才進 `evidence/runpod/<run_id>/`）

```
ssh -p <POD_SSH_PORT> root@<POD_HOST>      # placeholder；真實值不得出現在 repo
```

優先序：(1) FP8、AWQ 的 open-loop 掃描（policy (i)、無加速）→ r_SLO 與 $/M；(2) 預設精度的 25 min trace × 3 策略 × 3 seed。BF16 只在 ≤ 0.85 utilisation 裝得進時跑。不重跑 spec-decode 軸、TMMLU+、closed-loop 全掃描、GPTQ-Int4。

## 停止規則

- USD 10 或 10 GPU-h 先到者停；每日結束必刪 pod；任一阻塞錯誤 > 1 h 未解即停。
- 時價（console 當日）記入 `analysis/ledger/cost.csv`；GPU $/s = 時價 ÷ 3600；仍取樣功耗算 Wh/M。

## Secrets boundary

`make audit-secrets` 掃 IP、私鑰 header、SSH 公鑰、RunPod API key、RunPod host、HF token、email；pre-commit 與 CI 都跑。
