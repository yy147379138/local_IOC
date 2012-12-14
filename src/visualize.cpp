#include "visualize.h"
#include <cmath>
#include <cassert>

bool operator==(RGBTRIPLE &rgb1, RGBTRIPLE &rgb2) {
   return (rgb1.R == rgb2.R 
         && rgb1.G == rgb2.G 
         && rgb1.B == rgb2.B);
}

RGBTRIPLE operator*(RGBTRIPLE &rgbIn, double scale) {
   RGBTRIPLE rgb;
   rgb.R = (char)floor((unsigned char)rgbIn.R * scale);
   rgb.G = (char)floor((unsigned char)rgbIn.G * scale);
   rgb.B = (char)floor((unsigned char)rgbIn.B * scale);
   return rgb;
}

RGBTRIPLE operator+(RGBTRIPLE &rgbIn1, RGBTRIPLE &rgbIn2) {
   RGBTRIPLE rgb;
   rgb.R = rgbIn1.R + rgbIn2.R;
   rgb.G = rgbIn1.G + rgbIn2.G; 
   rgb.B = rgbIn1.B + rgbIn2.B;
   return rgb; 
}

RGBTRIPLE operator-(RGBTRIPLE &rgbIn1, RGBTRIPLE &rgbIn2) {
   RGBTRIPLE rgb;
   rgb.R = (unsigned int)rgbIn1.R - (unsigned int)rgbIn2.R;
   rgb.G = (unsigned int)rgbIn1.G - (unsigned int)rgbIn2.G; 
   rgb.B = (unsigned int)rgbIn1.B - (unsigned int)rgbIn2.B;
   return rgb; 
}


#if 0
void BMPFile::addBelief(vector<vector<double> > &values,
      double scale, double threshold, RGBTRIPLE colorConst,
      RGBTRIPLE colorScale, bool add) {
   for (int i=0; i < values.size(); i++) {
      for (int j=0; j < values.at(i).size(); j++) {
         double val = values.at(i).at(j) * scale;
         if (val < threshold) continue;
         //val = -0.0;
         double pixelW = (val+255.0)/255.00001;

         //pixelW = min(pixelW, 0.5);
         pixelW = max(pixelW, 0.0);
         //if (pixelW > 0.0)  cout << "PIXELW: "<<pixelW<<endl;
         //cout << "PIXELW: "<<pixelW<<"   "<<val<<endl;
         
         RGBTRIPLE colorWeighted = colorScale * pixelW;
         RGBTRIPLE color;
         if (add) 
            color = colorConst + colorWeighted;
         else 
            color = colorConst - colorWeighted;
         setPixel(i, j, color);
      }
   }
}
#else

void BMPFile::addBelief(vector<vector<double> > &values,
      double minVal, double maxVal, RGBTRIPLE colorConst,
      RGBTRIPLE colorScale, bool add) {

   pair<int, int> beliefDims;
   beliefDims.first = values.size();
   beliefDims.second = values.at(0).size();

   int scale = (int)ceil((float)xDim/beliefDims.first); 

   assert(scale == ceil((float)yDim/beliefDims.second));

   for (int i=0; i < values.size(); i++) {
      for (int j=0; j < values.at(i).size(); j++) {
         double val = values.at(i).at(j);

         double pixelW;
         if (val < minVal) 
            pixelW = 0.0;
         else if (val > maxVal) 
            pixelW = 1.0;
         else 
            pixelW = (val-minVal)/(maxVal-minVal);

         RGBTRIPLE color;

         int R1 = colorConst.R & 0xFF;
         int R2 = colorScale.R & 0xFF; 
         int G1 = colorConst.G & 0xFF;
         int G2 = colorScale.G & 0xFF;
         int B1 = colorConst.B & 0xFF;
         int B2 = colorScale.B & 0xFF;
 
         color.R = (char)floor((1.0-pixelW)*R1+pixelW*R2); 
         color.G = (char)floor((1.0-pixelW)*G1+pixelW*G2); 
         color.B = (char)floor((1.0-pixelW)*B1+pixelW*B2);

         setPixel(i, j, color, scale);
      }
   }
}


void BMPFile::addBelief(vector<vector<double> > &values,
      double minVal, double maxVal, ColorMap &cMap, bool add) {

   pair<int, int> beliefDims;
   beliefDims.first = values.size();
   beliefDims.second = values.at(0).size();

   int scale = (int)ceil((float)xDim/beliefDims.first); 

   assert(scale == ceil((float)yDim/beliefDims.second));

   for (int i=0; i < values.size(); i++) {
      for (int j=0; j < values.at(i).size(); j++) {
         double val = values.at(i).at(j);
         double pixelW;
         if (val < minVal) {
            pixelW = 0.0;
            //continue; 
         }
         else if (val > maxVal) 
            pixelW = 1.0;
         else 
            pixelW = (val-minVal)/(maxVal-minVal);

         RGBTRIPLE color;

         Color color2 = cMap.getColor(pixelW);

         int R1 = (int)floor(255*color2.getR()) & 0xFF;
         int G1 = (int)floor(255*color2.getG()) & 0xFF; 
         int B1 = (int)floor(255*color2.getB()) & 0xFF;
         
         color.R = (char)R1; 
         color.G = (char)G1;
         color.B = (char)B1;

         setPixel(i, j, color, scale);
      }
   }
}




#endif 
void BMPFile::addVector(vector<pair<int, int> > &positions, RGBTRIPLE color,
      int factor, int linesize) {
   for (int i=0; i < positions.size(); i++) {
      for (int x=1-linesize;x < linesize; x++) {
         for (int y=1-linesize;y < linesize; y++) {
            setPixel(positions.at(i).first+x, positions.at(i).second+y, 
                  color, factor);
         }
      }
   }
}

void BMPFile::addLabel(pair<int,int> &pos, RGBTRIPLE color, int radius,int type){
	if (type==CIRCLE){
		for (int x=max(pos.first-radius+1,0);x<=min(pos.first+radius-1,xDim-1);x++){
			int slack = (int)floor(sqrt(radius*radius-(x-pos.first)*(x-pos.first)));
			for (int y=max(pos.second-slack,0);y<=min(pos.second+slack,yDim-1);y++)
				setPixel(x,y,color,1);
		}
	}else if(type==SQUARE){
		for (int x=max(pos.first-radius+1,0);x<=min(pos.first+radius-1,xDim-1);x++){
			int slack = radius-abs(x-pos.first);
			for (int y=max(pos.second-slack+1,0);y<=min(pos.second+slack-1,yDim-1);y++)
				setPixel(x,y,color,1);
		}
	}
}
		
vector<pair<int, int> > BMPFile::find(RGBTRIPLE color) {

   vector<pair<int, int> > points;

   RGBTRIPLE black = {0,0,0};
   RGBTRIPLE white = {255, 255, 255};

   for (int i=0; i < xDim; i++) { 
      for (int j=0; j < yDim; j++) {
         RGBTRIPLE pixelColor = getPixel(i,j);

         if (pixelColor == color) {
            cout << "IN"<<endl;
            points.push_back(pair<int, int>(i,j)); 
         }
      }
   }
   return points;
}

Color JetColorMap::getColor(double val) {

   double scale = ((val-minV)/(maxV-minV));
   scale = min(max(scale, 0.0), 1.0);

   int step = (int)floor(8.0*scale);

   double r, g, b;

   switch (step) {
      case 0:
         r=g=0.0;
         b=.5+scale*4.0;
         break;
      case 1:
      case 2:
         r=0.0;
         g=(scale-.125)*4.0;
         b=1.0;
         break;
      case 3:
      case 4:
         r=(scale-.375)*4.0;
         g=1.0;
         b=1-r;
         break;
      case 5:
      case 6:
         r=1.0;
         g=1-(scale-.625)*4.0;
         b=0.0;
         break;
      case 7:
      case 8:
      default:
         r=1-(scale-.875)*4.0;
         g=b=0.0;
         break;
   }
   return Color(r,g,b);
}

double JetColorMap::getValue(RGBTRIPLE colorRGB) {
   return getValue(Color(colorRGB.R, colorRGB.G, colorRGB.B));
}
 
double JetColorMap::getValue(Color color) {
   double val = 0.0;
   double r=((unsigned int)color.getR()%256)/256.0;
   double g=((unsigned int)color.getG()%256)/256.0;
   double b=((unsigned int)color.getB()%256)/256.0;

#if 0
   cout << "CHECK: ";
   cout << r << "  "<<g<<"  "<<b<<endl;
#endif

   if (r >=.995 && g >= .995 && b >= .995) 
      return -2.0;
   if (r==0.0 && g==0.0 && b==0.0)
      return -1.0;
   else if (r==0.0 && g==0.0) {
      val = (b-.5)/4.0;
      val = max(val, 0.0);
   }
   else if (r==0.0 && b>=.995) {
      val = g/4.0+.125; 
   }
   else if (g>=.995) {
     val = r/4.0+.375;
   }
   else if (r>=.995 && b==0.0) {
     val = (1-g)/4.0+.625;
   }
   else if (g==0.0 && b==0.0) {
      val = (1-r)/4.0+.875;
   }

   return val;
}


const RGBTRIPLE black = {0,0,0}; 
const RGBTRIPLE white = {255,255,255}; 
const RGBTRIPLE red = {0, 0, 255}; 
const RGBTRIPLE blue = {255, 0, 0}; 
const RGBTRIPLE green = {0, 255, 0}; 
const RGBTRIPLE initialColor = {152, 49, 111}; 
const RGBTRIPLE currentColor = {213, 165, 181}; 
const RGBTRIPLE magenta = {255, 0, 255}; 
const RGBTRIPLE cyan = {255, 255, 0}; 
const RGBTRIPLE yellow = {0, 255, 255};

