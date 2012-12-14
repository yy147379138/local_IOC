#ifndef VISUALIZE_H__
#define VISUALIZE_H__
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

using namespace std;

class ColorMap;

#define PACK_WORD(addr, value)  \
   *(addr)   = (value)      & 0xFF, \
   *(addr+1) = (value >> 8) & 0xFF

#define PACK_DWORD(addr, value) \
   PACK_WORD(addr, (value & 0xFFFF)), \
   PACK_WORD(addr+2, ((value >> 16) & 0xFFFF))


typedef struct {
   short type;        /* Magic identifier            */
   int size;              /* File size in bytes          */
   short reserved1;
   short reserved2;
   int offset;            /* Offset to image data, bytes */
} BMPHEADER;

typedef struct {
   unsigned int size;               /* Header size in bytes      */
   int width,height;                /* Width and height of image */
   unsigned short int planes;       /* Number of colour planes   (1)  */
   unsigned short int bits;         /* Bits per pixel            (24) */
   unsigned int compression;        /* Compression type          (0)  */
   unsigned int imagesize;          /* Image size in bytes       */
   int xresolution,yresolution;     /* Pixels per meter          */
   unsigned int ncolours;           /* Number of colours         */
   unsigned int importantcolours;   /* Important colours         */
} BMPINFOHEADER;

typedef struct {
   char B;
   char G;
   char R;
//   char unused;
} RGBTRIPLE;


bool operator==(RGBTRIPLE &rgb1, RGBTRIPLE &rgb2); 

RGBTRIPLE operator*(RGBTRIPLE &rgb, double scale);


class BMPFile {
   public:
	   enum{SQUARE,CIRCLE};
      BMPFile(string filename) {

         FILE *file;
         file = fopen(filename.c_str(), "rb");

         if (!file) {
            cout << "BMP File "<<filename<<" not found!"<<endl;
            exit(0);
         }
         int res = fread((char*)&header, 54, 1, file);

         if (res < 0) {
            cout << "Read Failure: BMPFile"<<endl;
            exit(0);
         } 
         
         xDim = (*(header+18) & 0xFF) + ((*(header+19) & 0xFF) << 8);
         yDim = (*(header+22) & 0xFF) + ((*(header+23) & 0xFF) << 8);

         values = new RGBTRIPLE[xDim*yDim];

         int fullLine = ((xDim*3+3)/4)*4; 

         char *rawData = new char[fullLine*yDim];

         res = fread(rawData, fullLine*yDim, 1, file);

         if (res < 0) {
            cout << "Read Failure: BMPFile"<<endl;
            exit(0);
         }

         /* Only copy the non-padded part */
         for (int i=0; i < yDim; i++) 
            memcpy(values+i*xDim, rawData+fullLine*i, xDim*3);
         
         fclose(file);
         delete[](rawData);
      }

      BMPFile(int _xDim, int _yDim) : xDim(_xDim), yDim(_yDim) {

         *header = 'B';
         *(header+1) = 'M';
         
         PACK_DWORD(header+2, 54 + xDim*yDim*sizeof(RGBTRIPLE)); 
         PACK_WORD(header+6, 0);
         PACK_WORD(header+8, 0);
         PACK_DWORD(header+10, 54);
         PACK_DWORD(header+14, 40);
         PACK_DWORD(header+18, xDim);
         PACK_DWORD(header+22, yDim);
         PACK_WORD(header+26, 1);
         PACK_WORD(header+28, 24);
         PACK_DWORD(header+30, 0);
         PACK_DWORD(header+34, xDim*yDim*sizeof(RGBTRIPLE));
         PACK_DWORD(header+38, 0);
         PACK_DWORD(header+42, 0);
         PACK_DWORD(header+46, 0);
         PACK_DWORD(header+50, 0);

         values = new RGBTRIPLE[xDim*yDim];

         RGBTRIPLE white;
         white.R = 255;
         white.G = 255;
         white.B = 255; 

         for (int x=0; x < xDim; x++) 
            for (int y=0; y < yDim; y++) {
               setPixel(x,y,white); 

               RGBTRIPLE checkColor = getPixel(x,y);
               if (!(checkColor == white)) {
                  cout << "MAJOR PROBLEMS!!!"<<endl;
                  exit(0);
               }
            }
      }

      void setPixel(int x, int y, RGBTRIPLE color, int size) {
         if (!(size*x < xDim && size*y < yDim && size*x >= 0 && size*y >= 0)) 
            cout << "Set Pixel: Out or range, ignoring."<<endl;
         
         else {
            for (int i=0; i < size; i++) {
               for (int j=0; j < size; j++) {
                  values[(yDim-size*y-1+i)*xDim+(x*size+j)] = color;
               }
            }
         }
      }



      void setPixel(int x, int y, RGBTRIPLE color) {
         if (!(x < xDim && y < yDim && x >= 0 && y >= 0)) 
            cout << "Set Pixel: Out or range, ignoring."<<endl;
         
         else
            values[(yDim-y-1)*xDim+x] = color; 
      }

      ~BMPFile() {
         delete[] values;
      }

      RGBTRIPLE getPixel(int x, int y) {

         return values[(yDim-y-1)*xDim+x];
      }

      pair<int, int> getDims() { 
         return pair<int, int>(xDim, yDim);
      }

      void write(string filename) {

         FILE *file = fopen(filename.c_str(), "wb");
       
		 if (!file){
		    cout<<"Null file pointer"<<endl;
		 }

         cout << "WRITING: "<<filename<<endl;
		 
		 /*
		 int len = strlen((char *)&header);
         cout<<"header length: "<<len<<endl;
         for (int i=0;i<len;i++){
			 printf("%c \n",header[i]);
		 }*/
         //int res = fwrite((char *)&header, 54, sizeof(char), file);
	
         int res = fwrite((char *)&header, sizeof(char), 54, file);
		 //cout<<"number written: "<<res<<endl;

         if (res < 0) {
            cout << "Write Failure: BMPFile"<<endl;
            exit(0);
         }


         int fullLine = ((xDim*3+3)/4)*4;
         char *writebuf = new char[yDim*fullLine]; 
         char zerobuf[] = {0,0,0,0};
        

		 /* Write padded lines */
         for (int i=0; i < yDim; i++) 
            memcpy(writebuf+i*fullLine, values+i*xDim, 3*xDim); 

         res = fwrite((char *)(writebuf), 1, fullLine*yDim, file);
 
         if (res < 0) {
            cout << "Write Failure: BMPFile"<<endl;
            exit(0);
         }

     
         delete[](writebuf);

         fclose(file); 
      }

      void addBelief(vector<vector<double> > &values, double scale, 
            double threshold, RGBTRIPLE colorConst, RGBTRIPLE colorScale, 
            bool add=true);

      void addBelief(vector<vector<double> > &values, double scale, 
            double threshold, ColorMap &cMap, bool add=true);


      void addVector(vector<pair<int, int> > &positions, RGBTRIPLE color,
            int factor=1, int size=1);

      void addLabel(pair<int, int> &pos, RGBTRIPLE color,
				  int radius=5, int type=SQUARE);

      vector<pair<int, int> > find(RGBTRIPLE color); 
   protected:
      RGBTRIPLE *values;

      char header[54];
      int xDim;
      int yDim;
};

class Color {
   public:
      Color(double _r, double _g, double _b) : r(_r), g(_g), b(_b) { }
      void setRGB(double _r, double _g, double _b) {
         r = _r;
         g = _g;
         b = _b;
      }
      double getR() { return r; }
      double getG() { return g; }
      double getB() { return b; }
   protected:
      double r, g, b;
};

class ColorMap {
   public:
      virtual Color getColor(double val) = 0; 
      virtual double getValue(Color color) = 0;
};


class JetColorMap : public ColorMap {
   public:
      JetColorMap(double minValue = 0.0, double maxValue = 1.0) :
      minV(minValue), maxV(maxValue) { }
      Color getColor(double val);
      double getValue(Color color);
      double getValue(RGBTRIPLE color);
   protected:
      double minV, maxV;
};

extern const RGBTRIPLE black; 
extern const RGBTRIPLE white;
extern const RGBTRIPLE red;
extern const RGBTRIPLE blue;
extern const RGBTRIPLE green;
extern const RGBTRIPLE initialColor;
extern const RGBTRIPLE currentColor;
extern const RGBTRIPLE magenta;
extern const RGBTRIPLE cyan;
extern const RGBTRIPLE yellow; 


#endif
