import CoreGraphics
import Foundation
import Vision

// Lab positive-control helper for ANE rail experiments.
//
// This runs Apple's Vision person-segmentation request in a tight loop against
// synthetic images. It is intentionally not Gomoku evidence; use it to prove
// powermetrics/Instruments can see a known ANE-backed workload before comparing
// Gomoku Core ML candidates. Conservative usage is one process with
// --workers 1; launch several single-worker processes for stronger pressure.

struct Config {
    var workers = 4
    var seconds = 3600.0
    var width = 1024
    var height = 1024
    var quality = VNGeneratePersonSegmentationRequest.QualityLevel.accurate
}

func parseConfig() -> Config {
    var config = Config()
    var args = Array(CommandLine.arguments.dropFirst())
    while !args.isEmpty {
        let key = args.removeFirst()
        guard !args.isEmpty else { break }
        let value = args.removeFirst()
        switch key {
        case "--workers":
            config.workers = max(1, Int(value) ?? config.workers)
        case "--seconds":
            config.seconds = max(1.0, Double(value) ?? config.seconds)
        case "--width":
            config.width = max(64, Int(value) ?? config.width)
        case "--height":
            config.height = max(64, Int(value) ?? config.height)
        case "--quality":
            if value == "fast" {
                config.quality = .fast
            } else if value == "balanced" {
                config.quality = .balanced
            } else {
                config.quality = .accurate
            }
        default:
            continue
        }
    }
    return config
}

func makeImage(width: Int, height: Int, seed: UInt8) -> CGImage {
    let rowBytes = width * 4
    var pixels = [UInt8](repeating: 0, count: rowBytes * height)
    for y in 0..<height {
        for x in 0..<width {
            let offset = y * rowBytes + x * 4
            pixels[offset] = UInt8((x + Int(seed)) & 255)
            pixels[offset + 1] = UInt8((y * 3 + Int(seed)) & 255)
            pixels[offset + 2] = UInt8(((x ^ y) + Int(seed)) & 255)
            pixels[offset + 3] = 255
        }
    }

    let data = Data(pixels) as CFData
    let provider = CGDataProvider(data: data)!
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    return CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: rowBytes,
        space: colorSpace,
        bitmapInfo: bitmapInfo,
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    )!
}

let config = parseConfig()
let deadline = Date().addingTimeInterval(config.seconds)
let group = DispatchGroup()
let lock = NSLock()
var total = 0

print("ANE Vision furnace starting workers=\(config.workers) image=\(config.width)x\(config.height) seconds=\(Int(config.seconds)) quality=\(config.quality)")
fflush(stdout)

for workerID in 0..<config.workers {
    group.enter()
    DispatchQueue.global(qos: .userInitiated).async {
        let image = makeImage(width: config.width, height: config.height, seed: UInt8(workerID * 37))
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = config.quality
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8

        var local = 0
        while Date() < deadline {
            autoreleasepool {
                let handler = VNImageRequestHandler(cgImage: image, options: [:])
                do {
                    try handler.perform([request])
                    local += 1
                    if local % 10 == 0 {
                        lock.lock()
                        total += 10
                        let snapshot = total
                        lock.unlock()
                        if workerID == 0 {
                            print("vision_completed=\(snapshot)")
                            fflush(stdout)
                        }
                    }
                } catch {
                    print("worker=\(workerID) error=\(error)")
                    fflush(stdout)
                }
            }
        }

        let remainder = local % 10
        if remainder > 0 {
            lock.lock()
            total += remainder
            lock.unlock()
        }
        group.leave()
    }
}

group.wait()
print("ANE Vision furnace done total=\(total)")
