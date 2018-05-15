#include "isomut2_lib.h"

int main(int argc, char** argv)
{
    //cmdline args
    //parameters for haploid_cov estimation
    if(argc<7){
        printf("ERROR please provide 6 args \n windowsize (100000) \n shiftsize (50000)\n min_noise (0.1)\n");
        printf(" print_every_nth_base (100), \n base quality limit (30), \n output filename \n sample name \n");
        exit(1);
    }
    int ws=(int) strtol(argv[1],NULL,10);
    int shift=(int) strtol(argv[2],NULL,10);
    double min_noise=strtod(argv[3],NULL);
    int print_every_nth=(int) strtol(argv[4],NULL,10);
    int baseq_limit=(int) strtol(argv[5],NULL,10);
    char * output_filename = argv[6];
    int n_sample_names=argc-7;
    char** sample_names= (char**) malloc(n_sample_names * sizeof(char*));
    //int MAX_CHROM_LEN = 100;
    int i, j;
    i=0;
    for(i=0;i<n_sample_names;i++) sample_names[i]=argv[7+i];

    //define counting variables
    double dip_cov_total = 0;
    int dip_count = 0;
    double trip_cov_total = 0;
    int trip_count = 0;

    //define and initialize array for running average

    int rows = 5; //pos, cov, ref_freq, total_cov, est_num
    double **window_data = malloc(sizeof *window_data * ws);
    for (i=0;i<ws;i++)
    {
        window_data[i] = malloc(sizeof * window_data[i] * rows);
    }
    for(i=0;i<ws;i++)
    {
        for(j=0;j<rows;j++)
        {
            window_data[i][j]=-42;
        }
    }

    //define array for chrom storage
    char *chrom_array[ws];
    for (i=0;i<ws;i++) chrom_array[i] = (char*) malloc((100) * sizeof(char));

    //varaiables for reading a line
    char* line = NULL;
    size_t len = 0;
    ssize_t line_size;

    //print header
    // printf("#chrom\tpos\tcov\tBAF\n");

    int pointer_windowdata = 0;

    //loop over input lines
    while ((line_size = getline(&line, &len, stdin)) != -1) {
        //the pileup line structure for the line being read
        struct mplp my_mplp;
        init_mplp(&my_mplp);

        //build the struct from input line
        process_mplp_input_line(&my_mplp,line,line_size,baseq_limit,sample_names,1);

        //build window_data
        build_window_data(window_data, chrom_array, &pointer_windowdata, &my_mplp);

        //free the memory allocated by the struct
        free_mplp(&my_mplp);

        if(pointer_windowdata == ws)
        {
          shift_window(window_data, chrom_array, &pointer_windowdata, ws, rows, shift,
                       min_noise, &dip_cov_total, &dip_count, &trip_cov_total, &trip_count, print_every_nth);
        }
    }

    print_last_window(window_data, chrom_array, ws,
                      min_noise, &dip_cov_total, &dip_count, &trip_cov_total, &trip_count, print_every_nth);

    FILE *file;
    file = fopen(output_filename, "w");
    fprintf(file, "%f\t%d\t%f\t%d\n", dip_cov_total, dip_count, trip_cov_total, trip_count);
    fclose(file);

    //free resources
    free(line);
    for (i = 0; i < ws; i++) free(window_data[i]);
    free(window_data);
    return 0;
}
