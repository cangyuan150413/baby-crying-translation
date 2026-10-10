# baby-crying-translation 婴儿哭声翻译

## backend（哭声分析模型 + 实时服务）
- `crynet/features.py`：音频特征（音高、频谱质心、频带能量、节奏等，纯 numpy）
- `crynet/model.py`：随机森林分类器，区分 饿(hungry)/困(tired)/不适(discomfort)/疼痛(pain)/拍嗝(burp)
- 默认使用合成数据训练以便开箱即用；有真实标注数据（`dir/<标签>/*.wav`）时：`python train.py dir`
- 运行：`cd backend && pip install -r requirements.txt && uvicorn server:app --host 0.0.0.0`
  - `WS /ws/analyze` 发送 16kHz 单声道 PCM16 音频，返回分析结果
  - `POST /analyze` 原始 PCM16 请求体
  - `WS /ws/subscribe` 供 App 实时接收结果
- 测试：`cd backend && python -m pytest tests`

## mobile（iOS / Android 通用 App，Expo React Native）
`cd mobile && npm install && npx expo start`，在 App 中填入服务器 `IP:8000`，实时显示哭声原因。

> 注意：合成数据仅为演示，实际准确率需用真实标注婴儿哭声数据训练。
