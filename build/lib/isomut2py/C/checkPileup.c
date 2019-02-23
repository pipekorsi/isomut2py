#include "isomut2_lib.h"

int main(int argc, char** argv)
{
    //cmdline args
    //parameters for mutation calling
    if(argc<2){
        printf("ERROR please provide 2 args: \n\t");
        printf("baseq_limit (30),");
        printf("\n\tlist of sample names\n");
        exit(1);
    }
    int baseq_limit=(int) strtol(argv[1],NULL,10);
    int n_sample_names=argc-2;
    char** sample_names= (char**) malloc(n_sample_names * sizeof(char*));
    int i=0;
    for(i=0;i<n_sample_names;i++) sample_names[i]=argv[2+i];

    //varaiables for reading a line
    char* line = NULL;
    size_t len = 0;
    ssize_t line_size;

    //print header
    printf("#chr\tpos\tref");
    for (i=0;i<n_sample_names;i++){
      printf("\tcov_");
      printf("%s", sample_names[i]);
      printf("\tAfreq_");
      printf("%s", sample_names[i]);
      printf("\tCfreq_");
      printf("%s", sample_names[i]);
      printf("\tGfreq_");
      printf("%s", sample_names[i]);
      printf("\tTfreq_");
      printf("%s", sample_names[i]);
      printf("\tinsfreq_");
      printf("%s", sample_names[i]);
      printf("\tdelfreq_");
      printf("%s", sample_names[i]);
    }
    printf("\n");

    //process actual samtools mpileup lines:
    while ((line_size = getline(&line, &len, stdin)) != -1) {
        //the pileup line structure for the line being read
        struct mplp my_mplp;
        init_mplp(&my_mplp);

        //build the struct from input line
        process_mplp_input_line(&my_mplp,line,line_size,baseq_limit,sample_names,n_sample_names);

        //print detailed counts and frequencies
        print_mplp_basic_info(&my_mplp);

        //free the memory allocated by the struct
        free_mplp(&my_mplp);
    }


    //free resources
    free(line);
    return 0;
}
