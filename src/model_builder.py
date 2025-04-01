from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class ModelBuilder:
    def __init__(self, df, feature_cols, target_col='Target'):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col

    def preprocess(self):
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler

    def build_and_train(self):
        model = Sequential([
            Dense(64, activation='relu', input_dim=len(self.feature_cols)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=20, batch_size=32, validation_split=0.2)
        self.model = model
        return model

    def save(self, path='models/model.h5'):
        self.model.save(path)
