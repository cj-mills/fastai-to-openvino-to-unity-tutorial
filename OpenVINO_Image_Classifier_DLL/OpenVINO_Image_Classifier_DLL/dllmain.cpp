// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {

	// Inference engine instance
	ov::Core core;
	// The user define model representation
	std::shared_ptr<ov::Model> model;
	// A device-specific compiled model
	ov::CompiledModel compiled_model;
	
	// List of available compute devices
	std::vector<std::string> available_devices;
	// An inference request for a compiled model
	ov::InferRequest infer_request;
	// Stores the model input data
	ov::Tensor input_tensor;
	// A pointer for accessing the input tensor data
	float* input_data;
	
	// The number of image classes the current model can detect
	int num_classes = 0;
	// The current input image width
	int input_w;
	// The current input image height
	int input_h;
	// The total number pixels in the input image
	int nPixels;
	// The number of color channels in the input image
	int num_channels = 3;

	/// <summary>
	/// Get the number of available compute devices
	/// </summary>
	/// <returns></returns>
	DLLExport int GetDeviceCount() {

		// Reset list of available compute devices
		available_devices.clear();

		// Populate list of available compute devices
		for (std::string device : core.get_available_devices()) {
			// Skip GNA device
			if (device.find("GNA") == std::string::npos) {
				available_devices.push_back(device);
			}
		}
		// Return the number of available compute devices
		return available_devices.size();
	}

	/// <summary>
	/// Get the name of the compute device name at the specified index
	/// </summary>
	/// <param name="index"></param>
	/// <returns></returns>
	DLLExport std::string* GetDeviceName(int index) {
		return &available_devices[index];
	}

	
	/// <summary>
	/// Load a model from the specified file path
	/// </summary>
	/// <param name="modelPath">The path to the OpenVINO IR model file</param>
	/// <param name="index">The compute device index</param>
	/// <param name="inputDims">The source image resolution</param>
	/// <returns></returns>
	DLLExport int LoadModel(char* modelPath, int index, int inputDims[2]) {

		// Initialize return value
		int return_val = 0;
		// Specify the cache directory for compiled gpu models
		core.set_property("GPU", ov::cache_dir("cache"));

		// Try loading the specified model
		try { model = core.read_model(modelPath); } 
		// Return 1 if the model fails to load
		catch (...) { return 1; }

		// Try updating the model input dimensions
		try { model->reshape({ 1, 3, inputDims[1], inputDims[0] }); }
		// Return a value of 2 if we can't update the model input dimensions
		catch (...) { return_val = 2; }

		// Compile the loaded model for the target compute device
		auto compiled_model = core.compile_model(model, "MULTI",
			ov::device::priorities(available_devices[index]),
			ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
			ov::hint::inference_precision(ov::element::f32));

		// Get the number of classes the current model can detect
		ov::Output<const ov::Node> output = compiled_model.output();
		num_classes = output.get_shape()[1];
		// Create an inference request to use the compiled model
		infer_request = compiled_model.create_infer_request();

		// Get input tensor by index
		input_tensor = infer_request.get_input_tensor(0);

		// Get model input dimensions
		input_w = input_tensor.get_shape()[3];
		input_h = input_tensor.get_shape()[2];
		nPixels = input_w * input_h;

		// Get a pointer to the input tensor
		input_data = input_tensor.data<float>();
		
		// Return a value of 0 if the model loads successfully
		return return_val;
	}

	/// <summary>
	/// Perform inference with the provided texture data
	/// </summary>
	/// <param name="inputData"></param>
	/// <returns></returns>
	DLLExport int PerformInference(uchar* inputData) {
		
		// Initialize predicted class index to an invalid value
		int class_idx = -1;
				
		try {

			// Store the pixel data for the source input image in an OpenCV Mat
			cv::Mat texture = cv::Mat(input_h, input_w, CV_8UC4, inputData);
			// Remove the alpha channel
			cv::cvtColor(texture, texture, cv::COLOR_RGBA2RGB);

			// Iterate over each pixel in image
			for (int p = 0; p < nPixels; p++) {
				// Iterate over each color channel for each pixel in image
				for (int ch = 0; ch < num_channels; ++ch) {
					input_data[ch * nPixels + p] = texture.data[p * num_channels + ch] / 255.0f;
				}
			}

			// Perform inference
			infer_request.infer();

			// model has only one output
			ov::Tensor output_tensor = infer_request.get_output_tensor();
			// IR v10 works with converted precisions (i64 -> i32)
			auto out_data = output_tensor.data<float>();

			// Get the predicted class index with the highest confidence score
			class_idx = std::distance(out_data, std::max_element(out_data, out_data + num_classes));
		}
		catch (...) {
			// Return a value of -2 if an error occurs during the forward pass
			class_idx = -2;
		}
		
		return class_idx;
	}
}
