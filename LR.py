# BIL 470
# Odev 2
# Seyyid Hikmet Celik
# 181201047

class LinearRegression:
    def __init__(self, learning_rate=0.000005, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        
        self.m1 = 1
        self.m2 = 2
        self.b = 0

    def fit(self, x_train, y_train, z_train):
        loss_list = []
        n = len(z_train)
        for i in range(self.epoch):
            z_predicted = self.m1 * x_train + self.m2 * y_train + self.b
            error = z_predicted - z_train
            # mean square error
            loss = sum(error**2)/n

            # parcali turevler
            loss_m1 = 2 * sum(error * x_train)/n 
            loss_m2 = 2 * sum(error * y_train)/n
            loss_b = 2 * sum(error)/n

            # yeni denklem katsayilari
            self.m1 = self.m1 - self.learning_rate * loss_m1
            self.m2 = self.m2 - self.learning_rate * loss_m2
            self.b = self.b - self.learning_rate * loss_b
            print("loss: " + loss.__str__() + "  \t(" + (i+1).__str__() + "/" + self.epoch.__str__() + ")")
            loss_list.append(loss)
        
        return loss_list

    def predict(self, x_test, y_test):
        return self.m1 * x_test + self.m2 * y_test + self.b

    # test verilerinin lossunu elde etmek icin ekstra fonksiyon
    def get_loss(self, x, y, z):
        loss_list = []
        n = len(z)
        for i in range(self.epoch):
            z_predicted = self.m1 * x + self.m2 * y + self.b
            error = z_predicted - z
            loss = sum(error**2)/n
            
            print("loss: " + loss.__str__() + "  \t(" + (i+1).__str__() + "/" + self.epoch.__str__() + ")")
            loss_list.append(loss)
        
        return loss_list
