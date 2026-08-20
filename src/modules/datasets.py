import os
import pandas as pd
from numpy import unique
from torch.utils.data import Dataset 
from sklearn.preprocessing import LabelEncoder
from torch import from_numpy
  
class BAF(Dataset): 
    def __init__(self, variant, root='data/', train=None, mode='train', validation=False):
        if variant.lower() not in ["base", "typei", "typeii", "typeiii", "typeiv", "typev"]:
            raise ValueError("Invalid variant. Choose between Base, TypeI, TypeII, TypeIII, TypeIV, TypeV.")
        if mode.lower() not in ["train", "test", "validation"]:
            raise ValueError("Invalid mode. Choose between train, test, and validation.")
        if mode.lower() == "validation" and not validation:
            raise ValueError("Validation mode requires validation=True.")
        self.target_name = None
        self.features = None
        self.categorical_features = ["payment_type","employment_status","housing_status","source","device_os"]
        self.categorical_encoder = LabelEncoder()
        if train is None:
            self._mode = mode
        else:
            self._mode = 'train' if train else 'test'
        self._validation = validation
        self.data, self.targets = self._load_data(root, variant)
        self.data = from_numpy(self.data.values).float().unsqueeze(1)
        self.targets = from_numpy(self.targets.values).int()
        self.classes = unique(self.targets)
    
    def __len__(self): 
        return len(self.data) 
  
    def __getitem__(self, index): 
        return self.data[index], self.targets[index]
    
    def _get_classes_category(self):
        idx_client = [0, 2, 3, 4, 6, 7, 14, 15, 17, 18, 19, 20, 21, 22]
        idx_system = [1, 5, 8, 9, 10, 11, 12, 13, 16, 23, 24, 25, 26, 27, 28, 29, 30] 
        features_client = self.features[idx_client]
        features_system = self.features[idx_system]
        return features_client, features_system
    
    def _read_data(self, root, variant):
        dataset = pd.read_parquet(f"{root}/{variant}.parquet")
        self.features = dataset.columns[1:]
        self.target_name = dataset.columns[0]
        for feature in self.categorical_features:
            self.categorical_encoder.fit(dataset[feature]) 
            dataset[feature] = self.categorical_encoder.transform(dataset[feature])  
        train = dataset[dataset["month"]<=6]
        if self._validation:
            test = dataset[dataset["month"]==7]
            validation = dataset[dataset["month"]==8]
        else:
            test = dataset[dataset["month"]>6]
            validation = None
        return train, test, validation
    
    def _load_data(self, root, variant):
        train, test, validation = self._read_data(root, variant)
        if self._mode == "train":
            return train.drop(columns=["fraud_bool"]), train["fraud_bool"]
        elif self._mode == "validation":
            return validation.drop(columns=["fraud_bool"]), validation["fraud_bool"]
        else:
            return test.drop(columns=["fraud_bool"]), test["fraud_bool"]
        
if __name__ == "__main__":
    path = os.getcwd()
    dataset = BAF(variant="Base", root=path + '/data/BAF', train=True, mode='train', validation=False)
    print(f"Dataset loaded with {len(dataset)} samples and {len(dataset.classes)} classes.")
    print(f"Features: {dataset.features}")
    print(f"Client Features: {dataset._get_classes_category()[0]}")
    print(f"System Features: {dataset._get_classes_category()[1]}")
