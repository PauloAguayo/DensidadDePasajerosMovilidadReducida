from measurements import Measurements
from draw import Drawing

class Outcomes(object):
    def __init__(self,dec,wr,ws):
        self.decision = dec
        self.standard_area = wr*ws
        self.measures = Measurements(self.decision)

    def Extra_dec(self):
        if self.decision==2:
                print('-------------------------------------------------------------------------')
                self.standard_area = float(input('MEASURE OF THE POLYGON AREA? (GROUND TRUTH) ' ))

    def Mode(self):
        if self.decision==1:
            self.area_name = "POLYGON AREA CALCULATED ="
            print('-------------------------------------------------------------------------')
            print('----- WHEELCHAIR AREA AS GROUND TRUTH MEASURE -----')
            print('-------------------------------------------------------------------------')
            print('WHEELCHAIR AREA =', self.standard_area, 'm2')
            print('-------------------------------------------------------------------------')
        elif self.decision==2:
            self.area_name = "WHEELCHAIR AREA CALCULATED ="
            print('-------------------------------------------------------------------------')
            print('----- POLYGON AREA AS GROUND TRUTH MEASURE -----')
            print('-------------------------------------------------------------------------')

    def txt(self):
        return(self.area_name)

    def People_density(self,people,polygon_area,chair_area):
        total_area = self.measures.Area(polygon_area,chair_area,self.standard_area)
        if self.decision==1:
            people_density = people/total_area
            return([people_density,total_area])
        elif self.decision==2:
            people_density = people/self.standard_area
            return([people_density,total_area])
