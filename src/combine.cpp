#include "visualize.h"
#include "grid.h"
#include "options.h"
#include "evidence.h"
#include "inference.h"
#include "main.h"
#include "features.h"

int main(int argc, char **argv) {

#if 0 

   int offset = 40;

   int frame = 0;

   for (int i=0; i < offset; i++) {
      //printf("cp multi1/occup%03d.bmp combined/combineOut%03d.bmp\n",
      //      i, frame++);
      printf("../exec/combine.exe --inputs multi1/occup%03d.bmp "
            "multi2/occup%03d.bmp --output combined/combineOut%03d.bmp\n", 
            i, 0, frame++); 
   }

   for (int i=0; i <= 310; i++) {
      printf("../exec/combine.exe --inputs multi1/occup%03d.bmp "
            "multi2/occup%03d.bmp --output combined/combineOut%03d.bmp\n", 
            i+offset, i, frame++);
   }

   for (;frame<=514;frame++)
      printf("../exec/combine.exe --inputs multi1/occup%03d.bmp "
            "multi2/occup%03d.bmp --output combined/combineOut%03d.bmp\n", 
            frame, 310, frame); 

   return 0;
#endif
   OptionParser opts;

   vector<string> inputFiles;
   string outputFile;

   opts.addOption(new StringVectorOption("inputs", 
            "--inputs<filename>               : image inputs",
            "", inputFiles, true));

   opts.addOption(new StringOption("output", 
            "--output <filename>              : evidence file",
            "", outputFile, true));

   opts.parse(argc,argv);

   if (inputFiles.size() < 2) {
      cout << "Too few files"<<endl;
      exit(0);
   }

   BMPFile map("input/grid.bmp");
   Grid grid(map, black);

   BMPFile result(inputFiles.at(0));
   
   JetColorMap jet;

   // result.setRange(-30.0, 5.0)
 
   pair<int, int> dims = result.getDims();

   vector<vector<double> > values(dims.first, vector<double>(dims.second, -HUGE_VAL));


   vector<pair<int, int> > pointVec;

   for (int i=0; i < inputFiles.size(); i++) {
      BMPFile image(inputFiles.at(i));

      for (int x=0; x < dims.first; x++) {

         for (int y=0; y < dims.second; y++) {
            double value = jet.getValue(image.getPixel(x,y));
            if (value == -2)
               pointVec.push_back(pair<int,int>(x,y));
            if (value > 0.0) {

               RGBTRIPLE rgb = image.getPixel(x,y);
               //   cout << ((unsigned int)rgb.R)%256 << " "<<((unsigned char)rgb.G)%256<<" "<<((unsigned int)rgb.B)%256<<endl;
               //cout << x << " "<<y<<"        "<<value<<endl;
               value = value*35.0-30.0;
               values.at(x).at(y) = LogAdd(value, values.at(x).at(y));
            }
         }
      }
   }



   BMPFile newOutput(dims.first, dims.second);

   newOutput.addBelief(values, -30.0, 5.0, jet);// (outputFile);
   
   grid.addObstacles(newOutput, black);

   newOutput.addVector(pointVec, white);

   newOutput.write(outputFile);

   return 0;
}
