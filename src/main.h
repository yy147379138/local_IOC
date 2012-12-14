#ifndef MAIN_H__
#define MAIN_H__

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>

using namespace std;

void convert(string str, string delim, vector<int> &results);
#if 0
{
   int next;
   char buf[20];
   while ( (next=str.find_first_of(delim)) != str.npos) {
      if (next > 0)
         results.push_back(atoi(str.substr(0,next).c_str()));

      str = str.substr(next+1);
   } 
}
#endif

void convert(string str, string delim, vector<double> &results);
#if 0
{
   int next;
   char buf[20];
   while ( (next=str.find_first_of(delim)) != str.npos) {
      if (next > 0)
         results.push_back(atof(str.substr(0,next).c_str()));

      str = str.substr(next+1);
   }
}
#endif


#endif
