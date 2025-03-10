#include </torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


struct NeuralNet : torch::nn::Module {
    NeuralNet() {
        fc1 = register_module("fc1", torch::nn::Linear(45, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, 64));
        fc4 = register_module("fc4", torch::nn::Linear(64, 2));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({x.size(32), 45})));
        x = torch::dropout(x, 0.5, is_training());
        x = torch::relu(fc3->forward(x));
        x = fc4->forward(x);
        return x;
    }
    torch::nn::Linear fc1{nullptr}, fc3{nullptr}, fc4{nullptr};

};

class AmpDataset : public torch::data::Dataset<AmpDataset>{
    private:
    torch::Tensor features_;
    torch::Tensor labels_;

public:
    explicit AmpDataset(const std::string& feature_file, const std::string& label_file) {
        // Load features
        std::ifstream features_in(feature_file);
        std::vector<std::vector<double>> features;
        std::string line, value;
        int i=0;
        while (std::getline(features_in, line)) {
            i++;
            std::stringstream ss(line);
            std::vector<double> row;

            while (std::getline(ss, value, ',')) {
                try
                {
                    row.push_back(std::stof(value));
                }
                catch(const std::exception& e)
                {
                    std::cerr << "Error parsing value '" << value << "' on line " << i << std::endl;
                    continue;
                }
                
            }
            features.push_back(row);
        }

        // Flatten features and convert to tensor, tensrors expect only 1d data
        std::vector<float> flat_features;
        size_t num_features = features[0].size();
        
        for (const auto& row : features) {
            flat_features.insert(flat_features.end(), row.begin(), row.end());
        }
        features_ = torch::from_blob(flat_features.data(), {static_cast<long>(features.size()), static_cast<long>(num_features)}, torch::kFloat).clone();
        
        int j=0;
        std::ifstream labels_in(label_file);
        std::vector<int64_t> labels;
        while (std::getline(labels_in, line)) {
            j++;
            try
            {
                labels.push_back(std::stoi(line));
            }
            catch(const std::exception& e)
            {
                std::cerr << "Error parsing label '" << line << "' on line " << j << std::endl;
                continue;
            }
            
            
        }
        labels_ = torch::tensor(labels, torch::dtype(torch::kInt64));
    }

    torch::data::Example<> get(size_t index) override {

        return {features_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return features_.size(0);
    }
};

int main() {
    // File paths to processed features and labels
    std::string label_file= "/home/arjissuan/PycharmProjects/Master_Thesis/DeepLearning/data/ML_AMP_class_DP.csv";
    std::string feature_file = "/home/arjissuan/PycharmProjects/Master_Thesis/DeepLearning/data/ML_AMP_features_DP.csv";

    // Create the dataset
    auto dataset = AmpDataset(feature_file, label_file);
    auto data_loader = torch::data::make_data_loader(std::move(dataset), torch::data::DataLoaderOptions().batch_size(32));

    // Initialize the model
    auto model = std::make_shared<NeuralNet>();

    // Optimizer and loss
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
    auto criterion = torch::nn::CrossEntropyLoss();

    // Training loop
    for (size_t epoch = 0; epoch < 10; ++epoch) {

        for (auto& batch : *data_loader) {
            std::vector<torch::Tensor> batch_data;
            std::vector<torch::Tensor> batch_targets;

            // Combine examples into a single batch tensor
            for (const auto& example : batch) {
                batch_data.push_back(example.data);
                batch_targets.push_back(example.target);
            }

            // Stack data and target tensors
            auto data = torch::stack(batch_data);
            auto target = torch::stack(batch_targets);

            // Forward pass, loss, backward, and update
            auto output = model->forward(data);
            auto loss = torch::nn::functional::binary_cross_entropy_with_logits(output, target);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            std::cout << "Loss: " << loss.item<float>() << std::endl;
        }
        std::cout << "Epoch [" << epoch + 1 << "/10] completed.\n";
    }

    std::cout << "Training complete!" << std::endl;
    return 0;
}