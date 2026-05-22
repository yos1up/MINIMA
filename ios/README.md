# MinimaLoFTRDemo (iOS)

MINIMA-LoFTR をオンデバイスで動かす最小デモ。SwiftUI + ONNX Runtime Mobile。

このディレクトリには **Swift ソースだけ** が入っています。Xcode プロジェクト（`.xcodeproj`）は自分で作成して、これらのファイルをドラッグで取り込みます（理由は後述）。

## ファイル

| ファイル | 役割 |
|---|---|
| `MinimaLoFTRDemoApp.swift` | アプリのエントリポイント（`@main`） |
| `ContentView.swift` | UI：画像 2 枚を選択 → Match ボタン → 結果表示 |
| `LoFTRMatcher.swift` | ONNX セッションの管理＋全体オーケストレーション |
| `ImagePreprocess.swift` | `UIImage` → `(1,1,640,640)` float32 グレースケール |
| `CoarseMatching.swift` | dual softmax＋mutual NN＋border mask（CPU 側） |
| `FineWindowCrop.swift` | 5×5 ウィンドウ抽出＋coarse 特徴 gather |
| `MatchVisualizer.swift` | 2 枚並べてマッチ線を描画 |

## Phase 別の作業（初心者向け詳細手順）

各 Phase の終わりに「動くもの」が手元にあるようにしてあります。詰まったら一つ前の Phase に戻ってください。

---

### Phase A：Xcode 環境セットアップ & 実機に "Hello World" を流す（30分〜2時間）

> **開発機について**: Apple Silicon / Intel Mac どちらでも全工程同じです。Intel Mac の場合、iOS シミュレータでの推論が体感 3〜5 倍遅くなりますが（CPU エミュではなく x86_64 ネイティブで動くものの GPU/ANE 周りが弱い）、実機ビルドとデプロイは ARM64 で同一です。**性能評価は必ず実機で行ってください**（Phase D）。Intel Mac で Xcode 16 を使うには macOS Sonoma 14.5 以降が必要です。

#### A-1. Xcode をインストール

App Store で **Xcode** を検索してインストール（数 GB あるので時間がかかります）。

インストール後、Xcode を起動して **Settings → Locations → Command Line Tools** に最新版が選ばれていることを確認。

#### A-2. Apple ID で署名できるようにする

- Xcode → **Settings → Accounts → +（左下）→ Apple ID** で自分の Apple ID を追加。
- 無料の Apple ID で「Personal Team」が使えるようになるので、これで個人の iPhone に転送可能になります（年間 99 USD の Developer Program は不要）。
- ただし無料署名には制約があります：**アプリが 7 日で期限切れ**になり、再度 Xcode から実機にインストールし直す必要があります。同時に保持できる Personal Team 製アプリも 3 個までです。デモ用なら十分。

#### A-3. "Hello World" プロジェクトで実機転送を確認

「本物のプロジェクト」を作る前に、まずは空のテンプレで実機転送が通ることを必ず確認してください。これが通らない状態で先に進むと、後で原因切り分けが難しくなります。

1. Xcode → **File → New → Project**
2. **iOS → App** を選択 → Next
3. Product Name: `HelloIOS`、Interface: **SwiftUI**、Language: **Swift** → Next → 保存場所を選んで Create
4. プロジェクト navigator（左サイドバー）でプロジェクト名（青いアイコン）をクリック → **TARGETS → HelloIOS → Signing & Capabilities** → **Team** に自分の Apple ID を選択
5. iPhone を Mac に Lightning/USB-C で接続。iPhone 側で「このコンピュータを信頼しますか？」が出たら **信頼**
6. Xcode の上部、ターゲット選択欄（▶️ ボタンの右）をクリックして **自分の iPhone** を選択
7. iPhone 側で **設定 → 一般 → VPN とデバイス管理 → デベロッパ App** に自分の Apple ID が出たら **信頼**
8. Xcode で ▶️（Run）。初回はビルドに 1〜2 分かかります
9. iPhone に "Hello, world!" の画面が表示されれば成功 🎉

**詰まりやすいポイント**

- **"Failed to register bundle identifier"**: Bundle Identifier がユニークでない。プロジェクト navigator の Signing & Capabilities で `com.yourname.HelloIOS` のような独自の名前に変更。
- **"Could not launch"**: iPhone 側の「デベロッパ App を信頼」が未実行。設定 → 一般 → VPN とデバイス管理 から。
- **iOS バージョンの警告**: TARGETS → General → Minimum Deployments を iOS 16.0 にしておくと、この PoC（PhotosPicker 使用）の要件と合います。

---

### Phase B：本物のプロジェクトを作って ONNX Runtime と .onnx を取り込む（30分）

#### B-1. プロジェクト作成

1. Xcode → **File → New → Project** → iOS → App
2. Product Name: `MinimaLoFTRDemo`、Interface: **SwiftUI**、Language: **Swift**
3. 作成。**最初に Run できることを確認**（空のテンプレが iPhone 上で起動するか）
4. プロジェクト navigator → TARGETS → General → **Minimum Deployments → iOS 16.0**

#### B-2. ONNX Runtime Mobile を Swift Package Manager で追加

1. Xcode → **File → Add Package Dependencies…**
2. 右上の検索バーに次の URL を貼り付ける：

   ```
   https://github.com/microsoft/onnxruntime-swift-package-manager
   ```

3. **Dependency Rule** は `Up to Next Major Version` を選択（デフォルト）
4. **Add Package** → パッケージ製品の一覧から **`onnxruntime`**（または `onnxruntime-objc`）を選んで Target に追加 → Finish

これで `import onnxruntime_objc` が使えるようになります。

#### B-3. .onnx ファイルを書き出してプロジェクトに取り込む

このリポジトリの `mobile/export_onnx.py` を Mac 側で実行：

```bash
cd /path/to/MINIMA
git submodule update --init --recursive third_party/LoFTR_minima
bash weights/download.sh  # weights/minima_loftr.ckpt を取得
python -m mobile.export_onnx --ckpt weights/minima_loftr.ckpt --img-size 640 --out-dir mobile/onnx
```

`mobile/onnx/loftr_stage1_640.onnx`（約 56 MB）と `mobile/onnx/loftr_stage2.onnx`（約 1.6 MB）が生成されます。

Xcode で：

1. プロジェクト navigator の `MinimaLoFTRDemo` フォルダを右クリック → **Add Files to "MinimaLoFTRDemo"…**
2. 上記 2 ファイルを選択
3. ダイアログ：
   - ✅ **Copy items if needed**
   - ✅ **Add to targets: MinimaLoFTRDemo**
4. Add

両ファイルが青ではなく**黒**で表示され、右ペインの **Target Membership** にチェックが入っていれば取り込めています。

#### B-4. ビルドだけ通る状態にする

この時点で一度 ⌘B（Build）してエラーがないことを確認。`onnxruntime_objc` のリンクが通っているかどうかが分かります。

---

### Phase C：Swift ソースを取り込み、シミュレータで動作確認（1時間）

#### C-1. Swift ファイルをドラッグ

このリポジトリの `ios/MinimaLoFTRDemo/*.swift`（**7 ファイル**）を Xcode のプロジェクト navigator にドラッグ：

- `MinimaLoFTRDemoApp.swift`（既存のものを置き換える）
- `ContentView.swift`（同上）
- `LoFTRMatcher.swift`
- `ImagePreprocess.swift`
- `CoarseMatching.swift`
- `FineWindowCrop.swift`
- `MatchVisualizer.swift`

ダイアログでは：

- ✅ Copy items if needed
- ✅ Add to targets: MinimaLoFTRDemo

「既存ファイルを置き換えるか？」と聞かれたら Replace で OK（テンプレで作られた `*App.swift` と `ContentView.swift` は使いません）。

#### C-2. Info.plist にプライバシー説明を追加

写真ライブラリへのアクセスには説明文が必須です。

- プロジェクト navigator → TARGETS → MinimaLoFTRDemo → **Info** タブ
- **Custom iOS Target Properties** で右クリック → **Add Row**
- Key: `NSPhotoLibraryUsageDescription`
- Value: `Used to pick images for matching demo.`

#### C-3. シミュレータで起動して読み込みを確認

1. Xcode 上部のターゲット選択を **iPhone 15 (Simulator)** などに切り替え
2. ▶️ Run
3. シミュレータが起動 → アプリが立ち上がる
4. **写真ライブラリの確認**: シミュレータ → Safari でなにか画像を開いて長押し → 写真に保存（シミュレータの写真ライブラリは初期状態だと空）
5. アプリで Image 0 / Image 1 を選択 → **Match** ボタン
6. 30〜90 秒待つ（シミュレータは CPU のみで遅い）→ マッチ画像が表示されればここまで成功

シミュレータが極端に遅いのは正常です（特に Intel Mac では Stage 1 だけで 5〜15 秒かかることがあります）。実機で計測しましょう。シミュレータでの動作確認だけ早めたい場合は、暫定で `mobile/export_onnx.py --img-size 320` で 320px 版を作り、`LoFTRMatcher.imgSize = 320` に変更して取り込むのも手です（精度は落ちますが Stage 1 が約 1/4 のサイズになります）。

---

### Phase D：実機で動かす（1時間）

1. iPhone を接続、Xcode のターゲット選択で実機を選ぶ
2. ▶️ Run
3. 初回起動時、iPhone 側で写真へのアクセス許可ダイアログが出るので「すべての写真へのアクセスを許可」または「写真を選択」
4. Image 0 / Image 1 を選び Match
5. 画面下の status ラベルに各段の時間が出ます（例: `598 matches  ·  pre 80 / s1 1800 / coarse 240 / s2 30 = 2150 ms`）

**期待値の目安（iPhone 14 Pro 以降）**

- s1（Stage 1 ONNX）: 1500〜2500 ms（FP32, CPU EP）
- coarse（dual softmax + mutual NN）: 200〜400 ms
- s2（Stage 2 ONNX）: 10〜50 ms
- total: 2〜3 秒

これでも遅すぎる場合は次の Phase E に進みます。

---

### Phase E：最適化（CoreML EP / FP16）

#### E-1. CoreML execution provider を有効化

`LoFTRMatcher.swift` の `init` 内、`ORTSessionOptions` を作る箇所に追加：

```swift
let coreml = ORTCoreMLExecutionProviderOptions()
coreml.useCPUOnly = false
coreml.enableOnSubgraph = true
coreml.onlyEnableForDevicesWithANE = false
try opt.appendCoreMLExecutionProvider(with: coreml)
```

CoreML EP は Stage 1 の backbone（畳み込みの塊）を Neural Engine で実行できる場合に劇的に速くなります。`einsum` ノードは CPU フォールバックしますが、それでも Stage 1 が 300〜600 ms 程度に収まることが多いです。

#### E-2. FP16 に変換

Mac で：

```bash
pip install onnx onnxconverter-common
python - <<'EOF'
import onnx
from onnxconverter_common import float16
for name in ["loftr_stage1_640", "loftr_stage2"]:
    m = onnx.load(f"mobile/onnx/{name}.onnx")
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, f"mobile/onnx/{name}_fp16.onnx")
    print(name, "ok")
EOF
```

`keep_io_types=True` で外部 IO は FP32 のまま、内部演算だけ FP16。Xcode で `_fp16` 付きの .onnx に差し替えて再ビルド。

#### E-3. その他

- `setIntraOpNumThreads` の値を 2 / 4 / 8 で試して最良値を探す。
- 解像度を 480 に下げて再エクスポート → `loftr_stage1_480.onnx` を取り込み、`LoFTRMatcher.imgSize = 480` に変更すると s1 がさらに 30〜40% 速くなります（精度トレードオフ）。

---

## トラブルシューティング

| 症状 | 原因 / 対応 |
|---|---|
| `MatcherError.modelMissing("loftr_stage1_640")` | .onnx ファイルが Bundle に入っていない。Xcode で .onnx を選択 → File Inspector → **Target Membership** にチェック。 |
| `import onnxruntime_objc` で "No such module" | Phase B-2 の SPM 追加が未完了。File → Add Package Dependencies で URL を再投入。 |
| ビルド時に `Undefined symbol: cblas_sgemm` | `Accelerate.framework` がリンクされていない。通常は自動だが、念のため TARGETS → General → Frameworks にあるか確認。 |
| 実機で `Could not launch... operation couldn't be completed` | iPhone 側でデベロッパ App を信頼していない。設定 → 一般 → VPN とデバイス管理 → 自分の Apple ID → 信頼。 |
| 画像ピッカーで写真を選んだあとに画面が真っ白 | iOS 17 で起こることあり。`PhotosPickerItem.loadTransferable(type: Data.self)` が `nil` を返している可能性。HEIC 画像で再現する場合、JPEG に変換してから試す。 |
| マッチ数が 0 | 入力画像が真っ黒（前処理失敗）または極端に重ね合わせのないペア。`mobile/test_parity.py` で参照を取って比較。 |

## 数値的な確認

`mobile/test_parity.py` で PyTorch 参照と ONNX パイプラインが完全に一致することを Mac 側で確認しています（座標差 max 1e-4 px、信頼度差 max 1e-5）。Swift 側の `CoarseMatching` も同じアルゴリズムなので、もし結果が `mobile/test_parity.py` と大きく違うようなら、まず入力画像の前処理（リサイズ・スケール）を疑ってください。

## ライセンス

Apache-2.0（親リポジトリと同じ）。
