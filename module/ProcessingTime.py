class ProcessingTime:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_2 = 0
        self.avg = 0
        self.stddev = 0
        self.size = 0
        self.avg_size = 0

    def add(self, t):
        self.count=self.count+1
        self.sum = self.sum+t
        self.sum_2 = self.sum_2+t*t

    def add(self, t, s):
        self.count=self.count+1
        self.sum = self.sum+t
        self.sum_2 = self.sum_2+t*t
        self.size = self.size+s

    def update(self):
        try:
            self.avg_size = self.size / self.count
            self.avg = self.sum/self.count
            self.stddev = (self.sum_2-self.avg*self.sum)/(self.count-1)
        except ZeroDivisionError:
            self.avg = 0
            self.stddev = 0
            self.avg_size = 0

    def print(self):
        return "total="+str(self.count)+",size="+str(self.avg_size)+",avg="+str(self.avg)+",stddev="+str(self.stddev)