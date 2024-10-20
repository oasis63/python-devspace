import json
import re


class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
    
  def printPerson(self):
  	print(self.name, self.age)

p1 = Person("John", 36)

print(p1.name)
print(p1.age)
p1.printPerson()



class Student(Person):
    def __init__(self,name,age,year):
	    # Person.__init__(self,name,age)
         super().__init__(name, age)
         self.graduationyear = year


  
s1 = Student("Stud", 12,2019)
s1.printPerson()

print(s1)


l1 = ['faslj', 4324,'wowo']
print(json.dumps(l1))


txt = "The rain in Spain"
x = re.search("^The.*Spain$", txt)

if x:
  print("YES! We have a match!")
else:
  print("No match")