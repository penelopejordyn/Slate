// CardBackgroundConfigView.swift provides the UI for configuring card backgrounds,
// including selection between different background types and their settings.
import SwiftUI
import PhotosUI

// MARK: - Background Selection Sheet

struct CardBackgroundConfigView: View {
    @ObservedObject var card: ObservableCard
    @Environment(\.dismiss) var dismiss

    @State private var selectedBackgroundType: BackgroundType = .none
    @State private var showingImagePicker = false

    enum BackgroundType: String, CaseIterable, Identifiable {
        case none = "None"
        case solidColor = "Solid Color"
        case image = "Image"
        case lined = "Lined"
        case grid = "Grid"

        var id: String { rawValue }

        var icon: String {
            switch self {
            case .none: return "xmark.circle"
            case .solidColor: return "paintpalette.fill"
            case .image: return "photo.fill"
            case .lined: return "text.alignleft"
            case .grid: return "grid"
            }
        }
    }

    var body: some View {
        NavigationView {
            List {
                Section("Background Type") {
                    ForEach(BackgroundType.allCases) { type in
                        Button(action: {
                            selectBackgroundType(type)
                        }) {
                            HStack {
                                Image(systemName: type.icon)
                                    .foregroundColor(.blue)
                                    .frame(width: 30)
                                Text(type.rawValue)
                                Spacer()
                                if selectedBackgroundType == type {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .foregroundColor(.primary)
                    }
                }

                // Configuration for current background type
                if selectedBackgroundType == .lined {
                    LinedBackgroundConfigSection(card: card)
                } else if selectedBackgroundType == .grid {
                    GridBackgroundConfigSection(card: card)
                } else if selectedBackgroundType == .image {
                    ImageBackgroundConfigSection(card: card, showingImagePicker: $showingImagePicker)
                } else if selectedBackgroundType == .solidColor {
                    SolidColorConfigSection(card: card)
                }
            }
            .navigationTitle("Card Background")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .onAppear {
            // Set initial selected type based on current background
            selectedBackgroundType = getCurrentBackgroundType()
        }
        .sheet(isPresented: $showingImagePicker) {
            ImagePicker(card: card)
        }
    }

    private func getCurrentBackgroundType() -> BackgroundType {
        switch card.background.style {
        case .none:
            return .none
        case .solidColor:
            return .solidColor
        case .image:
            return .image
        case .lined:
            return .lined
        case .grid:
            return .grid
        }
    }

    private func selectBackgroundType(_ type: BackgroundType) {
        selectedBackgroundType = type

        switch type {
        case .none:
            card.background = CardBackground(style: .none)
        case .solidColor:
            card.background = CardBackground(style: .solidColor(color: .blue))
        case .image:
            if card.background.imageData == nil {
                // Show image picker if no image is set
                showingImagePicker = true
            }
            card.background.style = .image
        case .lined:
            card.background = .linedPaper()
        case .grid:
            card.background = .gridPaper()
        }
    }
}

// MARK: - Lined Background Configuration

struct LinedBackgroundConfigSection: View {
    @ObservedObject var card: ObservableCard

    var body: some View {
        Section("Line Settings") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Line Spacing: \(Int(card.background.spacing))pt")
                    .font(.subheadline)
                Slider(value: $card.background.spacing, in: 15...50, step: 1)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Line Width: \(String(format: "%.1f", card.background.lineWidth))pt")
                    .font(.subheadline)
                Slider(value: $card.background.lineWidth, in: 0.5...3.0, step: 0.1)
            }
        }

        Section("Margins") {
            MarginControl(title: "Left Margin", margin: $card.background.margins.left)
            MarginControl(title: "Right Margin", margin: $card.background.margins.right)
            MarginControl(title: "Top Margin", margin: $card.background.margins.top)
            MarginControl(title: "Bottom Margin", margin: $card.background.margins.bottom)
        }
    }
}

// MARK: - Grid Background Configuration

struct GridBackgroundConfigSection: View {
    @ObservedObject var card: ObservableCard

    var body: some View {
        Section("Grid Settings") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Cell Size: \(Int(card.background.spacing))pt")
                    .font(.subheadline)
                Slider(value: $card.background.spacing, in: 10...50, step: 1)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Line Width: \(String(format: "%.1f", card.background.lineWidth))pt")
                    .font(.subheadline)
                Slider(value: $card.background.lineWidth, in: 0.5...3.0, step: 0.1)
            }
        }

        Section("Margins") {
            MarginControl(title: "Left Margin", margin: $card.background.margins.left)
            MarginControl(title: "Right Margin", margin: $card.background.margins.right)
            MarginControl(title: "Top Margin", margin: $card.background.margins.top)
            MarginControl(title: "Bottom Margin", margin: $card.background.margins.bottom)
        }
    }
}

// MARK: - Image Background Configuration

struct ImageBackgroundConfigSection: View {
    @ObservedObject var card: ObservableCard
    @Binding var showingImagePicker: Bool

    var body: some View {
        Section("Image Settings") {
            if card.background.imageData != nil {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Opacity: \(Int(card.background.imageOpacity * 100))%")
                        .font(.subheadline)
                    Slider(value: $card.background.imageOpacity, in: 0.1...1.0, step: 0.05)
                }

                Button(action: {
                    showingImagePicker = true
                }) {
                    HStack {
                        Image(systemName: "photo.badge.plus")
                        Text("Change Image")
                    }
                }
            } else {
                Button(action: {
                    showingImagePicker = true
                }) {
                    HStack {
                        Image(systemName: "photo.badge.plus")
                        Text("Select Image")
                    }
                }
            }
        }
    }
}

// MARK: - Solid Color Configuration

struct SolidColorConfigSection: View {
    @ObservedObject var card: ObservableCard

    private let presetColors: [(name: String, color: CodableColor)] = [
        ("White", CodableColor(red: 1.0, green: 1.0, blue: 1.0)),
        ("Light Gray", .lightGray),
        ("Yellow", CodableColor(red: 1.0, green: 0.95, blue: 0.7)),
        ("Pink", CodableColor(red: 1.0, green: 0.8, blue: 0.9)),
        ("Light Blue", CodableColor(red: 0.8, green: 0.9, blue: 1.0)),
        ("Mint", CodableColor(red: 0.7, green: 1.0, blue: 0.9))
    ]

    var body: some View {
        Section("Color") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 80))], spacing: 12) {
                ForEach(presetColors, id: \.name) { preset in
                    Button(action: {
                        card.background.style = .solidColor(color: preset.color)
                    }) {
                        VStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(preset.color.color)
                                .frame(height: 60)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                                )
                            Text(preset.name)
                                .font(.caption)
                                .foregroundColor(.primary)
                        }
                    }
                }
            }
            .padding(.vertical, 8)
        }
    }
}

// MARK: - Margin Control

struct MarginControl: View {
    let title: String
    @Binding var margin: MarginConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(title, isOn: $margin.isEnabled)

            if margin.isEnabled {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Width: \(Int(margin.percentage))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Slider(value: $margin.percentage, in: 5...30, step: 1)
                }
                .padding(.leading, 20)
            }
        }
    }
}

// MARK: - Image Picker

struct ImagePicker: UIViewControllerRepresentable {
    @ObservedObject var card: ObservableCard
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1

        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.dismiss()

            guard let provider = results.first?.itemProvider else { return }

            if provider.canLoadObject(ofClass: UIImage.self) {
                provider.loadObject(ofClass: UIImage.self) { image, error in
                    DispatchQueue.main.async {
                        guard let uiImage = image as? UIImage else { return }

                        // Compress and convert to PNG data
                        if let imageData = uiImage.pngData() {
                            self.parent.card.background.imageData = imageData
                            self.parent.card.background.style = .image
                        }
                    }
                }
            }
        }
    }
}

// MARK: - ObservableCard Wrapper

class ObservableCard: ObservableObject {
    let card: Card

    @Published var background: CardBackground {
        didSet {
            card.background = background
        }
    }

    init(card: Card) {
        self.card = card
        self.background = card.background
    }
}
