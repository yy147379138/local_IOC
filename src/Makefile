###########################################################
### Makefile for building all the People Prediction stuff
###           Brian D. Ziebart (01/26/2009)
###########################################################
# changed by Derek Zhao, 03/04/2012 on Ubuntu 11.01 machine
# Added velocity state into MDP based model
# Added continuous linear-quadratic model 
# Replaced the GSL minimizer by self-written optimizer,08/12/2012

#coreobjects = grid.o evidence.o features.o inference.o visualize.o \
#				  optimization.o

#coreobjectsplus = grid.o evidence.o features.o inference.o visualize.o \
#				  optimization.o inferenceplus.o trajoptimizerplus.o 


coreobjects = grid.o evidence.o features.o inference.o visualize.o \
                optimization.o inferenceplus.o 

lqobjects = grid.o evidence.o visualize.o linearquadratic.o

predictobjects = grid.o evidence.o features.o inference.o visualize.o \
                    inferenceplus.o linearquadratic.o intent.o

localobjects = grid.o evidence.o visualize.o linearquadratic.o\
                localOptimalIOC.o

#CFLAGS = -pg -g -Wno-deprecated
CFLAGS = -O3 -Wno-deprecated -ffast-math -msse  #-Wall -Wno-sign-compare -Wno-uninitialized

CC = g++

LIBS = -lgsl -lgslcblas

all: train.exe forecast.exe combine.exe trainplus.exe forecastplus.exe trainLQ.exe \
	lqforecast.exe trainLocal.exe localforecast.exe

MAPROOT = ..


$(MAPROOT)/exec/train.exe: train.o $(coreobjects)
	$(CC) ${CFLAGS} train.o $(coreobjects) -o $(MAPROOT)/exec/train.exe

$(MAPROOT)/exec/forecast.exe: forecast.o $(coreobjects)
	$(CC) ${CFLAGS} forecast.o $(coreobjects) -o $(MAPROOT)/exec/forecast.exe

$(MAPROOT)/exec/combine.exe: combine.o $(coreobjects) 
	$(CC) ${CFLAGS} combine.o $(coreobjects) -o $(MAPROOT)/exec/combine.exe 

$(MAPROOT)/exec/trainplus.exe: trainplus.o $(coreobjects) 
	$(CC) ${CFLAGS} trainplus.o $(coreobjects) -o $(MAPROOT)/exec/trainplus.exe 

$(MAPROOT)/exec/forecastplus.exe: forecastplus.o $(predictobjects) 
	$(CC) ${CFLAGS} forecastplus.o $(predictobjects) -o $(MAPROOT)/exec/forecastplus.exe
	
$(MAPROOT)/exec/trainLQ.exe: trainLQ.o $(lqobjects) 
	$(CC) ${CFLAGS} trainLQ.o $(lqobjects) -o $(MAPROOT)/exec/trainLQ.exe

$(MAPROOT)/exec/lqforecast.exe: lqforecast.o $(lqobjects) 
	$(CC) ${CFLAGS} lqforecast.o $(lqobjects) -o $(MAPROOT)/exec/lqforecast.exe

$(MAPROOT)/exec/trainLocal.exe: trainLocal.o $(localobjects) 
	$(CC) ${CFLAGS} trainLocal.o $(localobjects) -o $(MAPROOT)/exec/trainLocal.exe
	
$(MAPROOT)/exec/localforecast.exe: localforecast.o $(localobjects) 
	$(CC) ${CFLAGS} localforecast.o $(localobjects) -o $(MAPROOT)/exec/localforecast.exe

.cpp.o:
	@echo "" 
	@echo "--== $@ ==--  "
	@echo "----------------------------------------------------------------"
	@echo "[$?]"
	@echo "----------------------------------------------------------------"
	$(CC) -c $< ${CFLAGS}
	@echo "----------------------------------------------------------------"
	@echo ""

# allRawGPS.pdf: ../figures/allRawGPS.pdf

$(MAPROOT)/figures/%.pdf: $(MAPROOT)/figures/%.ps
	@echo "Converting PostScript to PDF"
	ps2pdf $< $@

$(MAPROOT)/evidence/%.routes: $(MAPROOT)/evidence/%.gpx \
    $(MAPROOT)/exec/gpsfilter.exe
	@echo "-------= Creating routes directory =-------"
	#mkdir "$(subst .gpx,,$<)"
	@echo "-------= Removing old routes! =-------"
	#rm "$(subst .gpx,,$<)"*.route
	$(MAPROOT)/exec/gpsfilter.exe --inputs $< \
            --outputDir $(subst .gpx,,$<)

skipfilter:
	touch $(MAPROOT)/evidence/*.routes


list:
	 for file in ../evidence/newData/*.route; do echo $file; done > dataList.lst

depend:
	makedepend -Y ./*.h *.cpp *.h

clean:
	rm *.o $(MAPROOT)/exec/gps.exe $(MAPROOT)/exec/displayevidence.exe \
               $(MAPROOT)/exec/gpsfilter.exe $(MAPROOT)/exec/webrouter.exe

cleanNetwork:
	rm ../networks/*.seg ../networks/*.con ../networks/*.ftr \
	   ../networks/*.map ../networks/*.ftm ../networks/*.rdn

tar:
	tar -cf ../ppp.tar *.h *.cpp Makefile \
		data/hires_training_corrected.trajectory \
		input/grid.bmp

count:
	wc -l *.h *.cpp

../figures/allRawGPS.ps: ../exec/displayevidence.exe $(GPSTRACES)
	../exec/displayevidence.exe --inputs $(GPSTRACES) \
                --output ../figures/allRawGPS.ps

train: ../exec/train.exe 
	../exec/train.exe --inputs $(EVIDENCE)

traceRoute: ../exec/displayWeighted.exe
	../exec/displayWeighted.exe --inputs $(ROUTETRACE)

traverse: ../exec/gps.exe
	../exec/gps.exe --execute traverse --inputs $(EVIDENCE)

filterall: ../exec/gpsfilter.exe
	../exec/gpsfilter.exe --inputs $(GPSTRACES) --cleanDist 0.01 \
	  --segmentTime 120

copyWeb: 
	/sbin/service httpd stop; \
	cp ../exec/webrouter.exe ../../public_html/test138.fcgi; \
	/sbin/service httpd start

%.pdf: ../figures/%.pdf;

%.ps: ../figures/%.ps;

%.exe: ../exec/%.exe;

# %.cpp: ../src/%.cpp

# DO NOT DELETE

# evidence.h->grid.h
./evidence.o: grid.h visualize.h
# feature.h->main.h->grid.h
./features.o: main.h grid.h visualize.h
# grid.h-> visualize.h
./grid.o: visualize.h
# infernce.h->main.h grid.h features.h evidence.h
./inference.o: main.h grid.h visualize.h features.h evidence.h
# optimization.h->main.h grid.h evidence.h features.h inference.h
./optimization.o: main.h grid.h visualize.h evidence.h features.h inference.h
# evidence.cpp->main.h evidence.h grid.h
evidence.o: main.h evidence.h grid.h visualize.h
# features.cpp->features.h
features.o: features.h main.h grid.h visualize.h
# forecast.cpp->main.h options.h grid.h evidence.h visualize.h features.h inference.h
forecast.o: main.h options.h grid.h visualize.h evidence.h features.h
forecast.o: inference.h
# grid.cpp -> grid.h main.h
grid.o: grid.h visualize.h main.h
# inference.cpp->inference.h
inference.o: inference.h main.h grid.h visualize.h features.h evidence.h
inferenceplus.o: inference.h main.h grid.h visualize.h evidence.h features.h
#main.o: main.h options.h grid.h visualize.h evidence.h features.h inference.h
# optimization.cpp->optimization.h
optimization.o: optimization.h main.h grid.h visualize.h evidence.h features.h inference.h
train.o: main.h options.h grid.h visualize.h evidence.h features.h
train.o: inference.h optimization.h
visualize.o: visualize.h
evidence.o: grid.h visualize.h
features.o: main.h grid.h visualize.h
grid.o: visualize.h
inference.o: main.h grid.h visualize.h features.h evidence.h
#trajoptimizerplus.o: optimization.h main.h grid.h visualize.h evidence.h features.h inference.h
trainplus.o: options.h visualize.h optimization.h main.h grid.h evidence.h
trainplus.o: features.h inference.h mymath.h
./linearquadratic.o: main.h mymath.h evidence.h grid.h visualize.h
linearquadratic.o: linearquadratic.h mymath.h main.h evidence.h grid.h visualize.h
trainLQ.o: options.h main.h visualize.h evidence.h grid.h linearquadratic.h mymath.h
lqforecast.o: options.h main.h visualize.h evidence.h grid.h linearquadratic.h mymath.h
#intent.h->main.h visualize.h inference.h linearquadratic.h
./intent.o: main.h visualize.h inference.h linearquadratic.h
intent.o: main.h visualize.h inference.h grid.h features.h evidence.h linearquadratic.h
intent.o: mymath.h 
forecastplus.o: main.h options.h grid.h visualize.h evidence.h features.h inference.h
forecastplus.o: intent.h linearquadratic.h 
./localOptimalIOC.o: main.h mymath.h grid.h linearquadratic.h
localOptimalIOC.o: localOptimalIOC.h linearquadratic.h main.h mymath.h grid.h visualize.h evidence.h 
trainLocal.o: options.h main.h visualize.h evidence.h grid.h localOptimalIOC.h mymath.h
localforecast.o: options.h main.h visualize.h evidence.h grid.h localOptimalIOC.h mymath.h
