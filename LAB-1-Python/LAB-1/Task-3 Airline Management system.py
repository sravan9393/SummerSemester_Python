import random

class Flight:
    def __init__(self, airline_name, flight_number):                                        #Default constructor for flight class
        self.airline_name = airline_name
        self.flight_number = flight_number

    def flight_display(self):                                                               #Displaying flight details
        print('Airlines : ', self.airline_name)
        print('Flight number : ', self.flight_number)


class Employee:                                                                             #Employee class
    def __init__(self, emp_id, emp_name, emp_age, emp_gender):                              #Employee class constructor
        self.emp_name = emp_name
        self.emp_age = emp_age
        self.__emp_id = emp_id
        self.emp_gender =  emp_gender
    def emp_display(self):                                                                  #Displaying Employee details
        print("Name of employee: ",self.emp_name)
        print('Employee id: ', self.__emp_id)
        print('Employee age: ',self.emp_age)
        print('Employee gender: ', self.emp_gender)

class Passenger:                                                                            #Passenger class
    def __init__(self):
        Passenger.__passport_number = input("Enter the passport number of the passenger: ") #Passport number is declared as private data member
        Passenger.name = input('Enter name of the passenger: ')
        Passenger.age = input('Enter age of passenger : ')
        Passenger.gender = input('Enter the gender: ')
        Passenger.class_type = input('Select business or economy class: ')

class Baggage():                                                                            #Baggage class
    cabin_bag = 1
    bag_fare = 0
    def __init__(self, checked_bags):
        self.checked_bags = checked_bags
        if checked_bags > 2 :                                                               #Calculating the cost if there are more than two cabin bags
            for i in checked_bags:
                self.bag_fare += 100
        print("Number of checked bags allowed: ",checked_bags,"bag fare: ",self.bag_fare)


class Fare(Baggage):                                                                        #Fare class which is subclass of Baggage
    counter = 150                                                                           #Cost is fixed for purchasing at counter
    online = random.randint(110, 200)                                                       #Cost varies with ticket is purchased through online and fair is generated through random function
    total_fare=0
    def __init__(self):
        super().__init__(2)                                                                 #Super call
        x = input('Buy ticket through online or counter: ')
        if x == 'online':
            Fare.total_fare = self.online + self.bag_fare
        elif x == 'counter':
            Fare.total_fare = self.counter + self.bag_fare
        else:
            x=input('Enter correct transaction type:')
        print("Total Fare before class type: ",Fare.total_fare)


class Ticket(Passenger, Fare):                                                             #Multiple inheritence
    def __init__(self):
        print("Passenger name:",Passenger.name)                                            #Acccessing parent class variable
        if Passenger.class_type == "business":
            Fare.total_fare+=100
        else:
            pass
        print("Passenger class type:",Passenger.class_type)
        print("Total fare:",Fare.total_fare)                                              #Displaying total fair for itenary


f1=Flight('American Airlines',9999)
f1.flight_display()

emp1 = Employee('e1', 'emp_Deepika', 21, 'F')
emp1.emp_display()

p1 = Passenger()

fare1=Fare()

t= Ticket()