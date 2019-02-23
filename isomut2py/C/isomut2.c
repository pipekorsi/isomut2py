#include "isomut2_lib.h"

#define MUT_BUFFER_SIZE 128  //must be longer than the proximal gap filtering distance!

int main(int argc, char** argv)
{
    //cmdline args
    //parameters for mutation calling
    if(argc<10){
        printf("ERROR please provide 8 args: \n\t min_sample_freq (0.3), \n\t min_other_ref_freq (0.9),\n\t");
        printf(" cov_limit (10),\n\t baseq_limit (30),\n\t prox_gap_dist_SNV  (10),\n\t  prox_gap_dist_indel  (100),");
        printf("\n\t constant ploidy (2),");
        printf("\n\t ploidy info filepath ('no_ploidy_info'),");
        printf("\n\t uniqe_mutation_only (1)");
        printf("\n\t shared_by_all (1)");
        printf(" list of sample names\n");
        exit(1);
    }
    double min_sample_freq=strtod(argv[1],NULL);
    double min_cleanliness=strtod(argv[2],NULL);
    int cov_limit=(int) strtol(argv[3],NULL,10);
    int baseq_limit=(int) strtol(argv[4],NULL,10);
    int prox_gap_min_dist_SNV= (int) strtol(argv[5],NULL,10);
    int prox_gap_min_dist_indel= (int) strtol(argv[6],NULL,10);
    //default constant ploidy
    int default_ploidy=(int) strtol(argv[7],NULL,10);
    //load file with information about sample ploidy bed files
    char *ploidy_info_filepath = (char*) malloc(1000*sizeof(char));
    strcpy(ploidy_info_filepath, argv[8]);
    //only process unique mutations
    int unique_only=(int) strtol(argv[9],NULL,10);
    //skip mutations shared by all samples
    int shared_by_all=(int) strtol(argv[10],NULL,10);
    //sample names
    int n_sample_names=argc-11;
    char** sample_names= (char**) malloc(n_sample_names * sizeof(char*));
    int i=0;
    int j;
    for(i=0;i<n_sample_names;i++) sample_names[i]=argv[11+i];

    //varaiables for reading a line
    char* line = NULL;
    size_t len = 0;
    ssize_t line_size;

    //potential mutation list
    struct mplp* potential_mutations = (struct mplp*) malloc(MUT_BUFFER_SIZE * sizeof(struct mplp)) ;
    //init all
    for( i=0;i<MUT_BUFFER_SIZE;i++) init_mplp(&potential_mutations[i]);
    int mut_ptr = i = 0; //pointer for potential mutation buffer

    //variables for proximal gap filtering
    int last_gap_pos_start,last_gap_pos_end;
    last_gap_pos_start = last_gap_pos_end = -42;
    char* last_gap_chrom = NULL;
    int is_gap = 1 ; // 0 yes, 1 no
    int found_SNV = 0;

    //load file that defines path to ploidy region file for each sample (structure: file_path\tcomma_sep_list_of_sample_names)
    int number_of_ploidy_files = 0;
    int max_number_of_ranges = 0;

    //initializing array to store ploidyID (which ploidy file a sample belongs to), current_ploidy and next_pos_to_check for each sample
    int * ploidy_id = (int *)malloc(n_sample_names * sizeof(int));
    int * current_ploidy = (int *)malloc(n_sample_names * sizeof(int));
    int * next_pos_to_check = (int *)malloc(n_sample_names * sizeof(int));

    for (i=0;i<n_sample_names;i++)
    {
        ploidy_id[i]=-42; //meaning they do not have ploidy information
        current_ploidy[i]=default_ploidy; //default ploidy is diploid
        next_pos_to_check[i]=INT_MAX; //never check
    }

    if (strcmp(ploidy_info_filepath,"no_ploidy_info") != 0)
    {
        //getting number of ploidy files available and the line count of the largest file
        get_basic_ploidy_info(ploidy_info_filepath, &number_of_ploidy_files, &max_number_of_ranges);
    }

    //if no ploidy info is available, set variables to 1 to avoid errors
    if (number_of_ploidy_files == 0 || max_number_of_ranges == 0)
    {
        number_of_ploidy_files = 1;
        max_number_of_ranges = 1;
    }

    //print header
    printf("#sample_name\tchr\tpos\ttype\tscore\tref\tmut\tcov\tmut_freq\tcleanliness\tploidy\n");


    //initializing a 2D array to store ploidy information (number_of_ploidy_files x max_number_of_ranges, each element is a ploidy struct)
    struct ploidy **array_of_ranges = (struct ploidy **)malloc(number_of_ploidy_files * sizeof(struct ploidy));
    for (i=0; i<number_of_ploidy_files; i++)
    {
        array_of_ranges[i] = (struct ploidy *)malloc(max_number_of_ranges * sizeof(struct ploidy));
        for (j=0;j<max_number_of_ranges;j++)
            init_ploidy(&array_of_ranges[i][j]);
    }

    //if actual information is available, fill up the array
    if (strcmp(ploidy_info_filepath,"no_ploidy_info") != 0)
    {
        build_ploidy_ranges_array(ploidy_info_filepath, array_of_ranges, ploidy_id, next_pos_to_check, sample_names, &n_sample_names);
    }

    while ((line_size = getline(&line, &len, stdin)) != -1) {
        //the pileup line structure for the line being read
        struct mplp my_mplp;
        init_mplp(&my_mplp);

        //build the struct from input line
        process_mplp_input_line(&my_mplp,line,line_size,baseq_limit,sample_names,n_sample_names);


        //call snvs with forward prox gap filtering
        found_SNV = call_snv_with_pl(potential_mutations,&mut_ptr,&my_mplp,
                 min_sample_freq,min_cleanliness,cov_limit,
                 last_gap_chrom,last_gap_pos_end,prox_gap_min_dist_SNV,
                 ploidy_id, next_pos_to_check, current_ploidy, &max_number_of_ranges, array_of_ranges,
                 unique_only, default_ploidy, shared_by_all);

        if (found_SNV == 0){ //if we did actually find an SNV (or a noisy sample), we don't bother looking for indels
          //call indels with forward prox gap filtering
          call_indel_with_pl(potential_mutations,&mut_ptr,&my_mplp,
                     min_sample_freq,min_cleanliness,cov_limit,
                     last_gap_chrom,last_gap_pos_start,last_gap_pos_end,
                     prox_gap_min_dist_SNV,prox_gap_min_dist_indel,
                     ploidy_id, next_pos_to_check, current_ploidy, &max_number_of_ranges, array_of_ranges,
                    unique_only, default_ploidy, shared_by_all);
        }


        //update last gap position seen
        update_last_gap(&my_mplp,&last_gap_chrom,&last_gap_pos_start,&last_gap_pos_end,&is_gap);

        //if a gap starts at this pos, delete in hindsight all potential mutations too close
        if( is_gap == 0 ) proximal_gap_hindsight_filter(potential_mutations,&mut_ptr,
                                                        last_gap_chrom,last_gap_pos_start,
                                                        prox_gap_min_dist_SNV,prox_gap_min_dist_indel);

        //if potential mut container is almost full: flush and print the accepted mutations
        if( mut_ptr > MUT_BUFFER_SIZE - 2 ){
            flush_accepted_mutations(potential_mutations,my_mplp.chrom,my_mplp.pos,
                                     &mut_ptr,prox_gap_min_dist_SNV,prox_gap_min_dist_indel);
        }

        //free the memory allocated by the struct
        free_mplp(&my_mplp);
    }

    //print mutations left at the very end
    for(i=0;i<mut_ptr;i++){
        print_mutation(&potential_mutations[i]);
        free_mplp(&potential_mutations[i]);
    }

    //free resources
    free(line);
    free(potential_mutations);
    for (i=0;i<number_of_ploidy_files;i++)
    {
      for (j=0;j<max_number_of_ranges;j++)
          free_ploidy(&array_of_ranges[i][j]);
      free(array_of_ranges[i]);
    }
    free(array_of_ranges);
    free(ploidy_id);
    free(current_ploidy);
    free(next_pos_to_check);
    return 0;
}
