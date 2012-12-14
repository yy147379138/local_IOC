/*******************************************************************
 ***  Functionality for parsing command line options
 ***            Brian D. Ziebart (4/27/07)
 *******************************************************************/ 

#ifndef OPTIONS_H__
#define OPTIONS_H__

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>

/* Virtual parent class */

using namespace std;

class Option {
   public:
      Option(string _name, string _desc, bool _required) 
          : name(_name), desc(_desc), required(_required) { }; 
      virtual ~Option() { }

      virtual int parse(int argc, char **argv, int pos) = 0; 
      string getName() { return name; }
      void help() { cout << "   "<<desc<<endl; }
      bool isRequired() { return required; }
      bool wasParsed() { return parsed; } 
      virtual void setDefault() = 0;
   protected:
      string name, desc;
      bool parsed, required;
};

/* String */

class StringOption : public Option {
   public:
      StringOption(string _name, string _desc, string _defval, string & _ret, 
           bool required=false) : Option(_name, _desc, required), ret(_ret), 
           defval(_defval) { }
      int parse(int argc, char **argv, int pos) {
         if (strncmp(argv[pos], "--", 2) || strcmp(argv[pos]+2, name.c_str())) 
            return pos;
         if (pos+1 < argc) 
            ret = argv[pos+1];
         parsed = true;
         return pos+2;
      }
      void setDefault() { ret = defval; }
   private:
      string &ret, defval;
};

/* Integer */

class IntOption : public Option {
   public:
      IntOption(string _name, string _desc, int _defval, int & _ret, 
           bool required=false) : Option(_name, _desc, required), ret(_ret), 
           defval(_defval) { }
      virtual int parse(int argc, char **argv, int pos) {
         if (strncmp(argv[pos], "--", 2) || strcmp(argv[pos]+2, name.c_str()))
            return pos;
         if (pos + 1 < argc)
            ret = atoi(argv[pos+1]);
         parsed = true;
         return pos+2;
      }
      void setDefault() { ret = defval; }
   private:
      int &ret, defval;
};

/* Boolean */

class BoolOption : public Option {
   public:
      BoolOption(string _name, string _desc, bool _defval, bool & _ret,
           bool required=false) : Option(_name, _desc, required), ret(_ret), 
           defval(_defval) { }
      virtual int parse(int argc, char **argv, int pos) {
         if (strncmp(argv[pos], "--", 2) || strcmp(argv[pos]+2, name.c_str()))
            return pos;
         parsed = true;
         ret = true;
         return pos+1;
      }
      void setDefault() { ret = defval; }
   private:
      bool &ret, defval;
};

/* Double */

class DoubleOption : public Option {
   public:
      DoubleOption(string _name, string _desc, double _defval, double & _ret,
           bool required = false) : Option(_name, _desc, required), ret(_ret), 
           defval(_defval) { }
      virtual int parse(int argc, char **argv, int pos) {
         if (strncmp(argv[pos], "--", 2) || strcmp(argv[pos]+2, name.c_str()))
            return pos;
         if (pos + 1 < argc)
            ret = atof(argv[pos+1]);
         parsed = true;
         return pos+2;
      }
      void setDefault() { ret = defval; }
   private:
      double &ret, defval;
};

/* Vector of strings */

class StringVectorOption : public Option {
   public:
      StringVectorOption(string _name, string _desc, string _defval, 
         vector<string> & _ret, bool required = false) 
          : Option(_name, _desc, required), ret(_ret), defval(_defval) { }
      virtual int parse(int argc, char **argv, int pos) { 
         if (strncmp(argv[pos], "--", 2) || strcmp(argv[pos]+2, name.c_str()))
            return pos; 
         int i;
         for (i=pos+1; i < argc; i++) {
            if (!strncmp(argv[i], "-", 1)) {
               break;
            }
            ret.push_back(argv[i]); 
         }
         parsed = true; 
         return i;
      }
      void setDefault() { ret.push_back(defval); }
   private:
      vector<string> &ret;
      string defval;
};

/* Option Parser class */

class OptionParser {
   public:
     void addOption(Option *opt) {
        options.push_back(opt);
     }

     void parse(int argc, char **argv) {
        int oldi, j;
        for (int i=1;i < argc; i++) {
           oldi=i;
           for (j=0; j < (int)options.size(); j++) {
              i=options.at(j)->parse(argc, argv, i); 
              if (oldi != i) {
                 i--;
                 break; 
              }
           }
           if ( j == (int)options.size() )  {
              cout << endl<<"Unknown option: "<< argv[i] <<endl;
              abort();
           } 
        } 
        for (int j=0; j < (int)options.size(); j++) {
           if (!options.at(j)->wasParsed() ) {
              if (options.at(j)->isRequired()) {
                 cout << "Missing required option: "
                      <<options.at(j)->getName()<<endl;
                 abort();
              }
              else { }
                 options.at(j)->setDefault();
           }
        }
     }

     void abort() {
        cout << endl << "Command Line Options"<<endl;
        for (int j=0; j < (int)options.size(); j++) {
           options.at(j)->help();
        }
        cout << endl << endl;
        exit(1);
     }

   private:
      vector<Option*> options;
};


#endif 
