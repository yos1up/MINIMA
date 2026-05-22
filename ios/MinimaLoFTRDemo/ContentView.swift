// Minimal UI: pick two images from the photo library, run the model, show the
// visualization. iOS 16+ (PhotosPicker).

import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var image0: UIImage?
    @State private var image1: UIImage?
    @State private var pick0: PhotosPickerItem?
    @State private var pick1: PhotosPickerItem?

    @State private var result: LoFTRMatchResult?
    @State private var resultImage: UIImage?
    @State private var status: String = "Pick two images to match."
    @State private var isRunning = false

    private let matcher: LoFTRMatcher? = {
        do { return try LoFTRMatcher() }
        catch { print("LoFTRMatcher init failed: \(error)"); return nil }
    }()

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                pickerBox(title: "Image 0", image: image0, selection: $pick0) { newItem in
                    Task { image0 = await loadImage(from: newItem) }
                }
                pickerBox(title: "Image 1", image: image1, selection: $pick1) { newItem in
                    Task { image1 = await loadImage(from: newItem) }
                }
            }
            .padding(.horizontal)

            Button(action: run) {
                if isRunning {
                    ProgressView()
                } else {
                    Text("Match")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(canRun ? Color.accentColor : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
            }
            .disabled(!canRun || isRunning)
            .padding(.horizontal)

            Text(status)
                .font(.footnote)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            ScrollView {
                if let resultImage {
                    Image(uiImage: resultImage)
                        .resizable()
                        .scaledToFit()
                        .padding()
                }
            }
        }
        .padding(.top, 12)
    }

    private var canRun: Bool {
        image0 != nil && image1 != nil && matcher != nil
    }

    @ViewBuilder
    private func pickerBox(title: String,
                           image: UIImage?,
                           selection: Binding<PhotosPickerItem?>,
                           onChange: @escaping (PhotosPickerItem?) -> Void) -> some View {
        PhotosPicker(selection: selection, matching: .images, photoLibrary: .shared()) {
            ZStack {
                Rectangle().fill(Color.gray.opacity(0.15)).cornerRadius(8)
                if let image {
                    Image(uiImage: image).resizable().scaledToFit().cornerRadius(8)
                } else {
                    VStack(spacing: 4) {
                        Image(systemName: "photo.on.rectangle")
                        Text(title).font(.caption)
                    }
                    .foregroundColor(.secondary)
                }
            }
            .frame(height: 140)
        }
        .onChange(of: selection.wrappedValue) { onChange($0) }
    }

    private func loadImage(from item: PhotosPickerItem?) async -> UIImage? {
        guard let item else { return nil }
        do {
            if let data = try await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                return img
            }
        } catch {
            print("loadImage failed: \(error)")
        }
        return nil
    }

    private func run() {
        guard let matcher, let image0, let image1 else { return }
        isRunning = true
        status = "Running…"
        Task.detached(priority: .userInitiated) {
            do {
                let r = try matcher.match(image0: image0, image1: image1)
                let viz = MatchVisualizer.render(image0: image0, image1: image1, matches: r.matches)
                await MainActor.run {
                    self.result = r
                    self.resultImage = viz
                    self.status = String(format:
                        "%d matches  ·  pre %.0f / s1 %.0f / coarse %.0f / s2 %.0f = %.0f ms",
                        r.matches.count, r.preprocessMs, r.stage1Ms, r.coarseMs, r.stage2Ms, r.totalMs)
                    self.isRunning = false
                }
            } catch {
                await MainActor.run {
                    self.status = "Error: \(error)"
                    self.isRunning = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
