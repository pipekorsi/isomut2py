#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>

///////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////

//Size limitations

//maximum sample limit, change it if you have more samples
#define MAXSAMPLE 1024
//maximum length of an indel
#define MAX_INDEL_LEN 128


//values for easier usage

//counts are stored in an array
//element indices are defined here
#define COV_IDX 0
#define REF_IDX 1
#define A_IDX 2
#define C_IDX 3
#define G_IDX 4
#define T_IDX 5
#define DEL_IDX 6
#define INS_START_IDX 7
#define DEL_START_IDX  8
#define READ_START_IDX 9
#define READ_END_IDX 10
#define MAX_IDX 11


// 0 coverage samples also have some kind of "frequency"
// should always check for this when working with the frequencies
#define ZERO_COV_FREQ -9999


// values for categorizing positions as diploid or triploid based on BAF (raw estimation of haploid coverage)
#define DIP_MIN_FREQ 0.48
#define DIP_MAX_FREQ 0.52
#define TRIP_MIN_FREQ_LOW 0.31
#define TRIP_MAX_FREQ_LOW 0.35
#define TRIP_MIN_FREQ_HIGH 0.65
#define TRIP_MAX_FREQ_HIGH 0.69
#define MAX_COV_FOR_EST 100
#define MIN_COV_FOR_EST 5

///////////////////////////////////////////////////////////////////////////
// Structures
////////////////////////////////////////////////////////////////////////////

/*
    the mpileup struct
*/
struct mplp
{
    //sample names for easier output
    char** sample_names;
    int n_sample_names;

    //raw data
    char* raw_line;

    //position data
    char* chrom;
    int pos;
    char ref_nuq;

    //sample data
    int n_samples;
    int ploidy;

    //sample level raw data
    int raw_cov[MAXSAMPLE];
    char* raw_bases[MAXSAMPLE];
    char* raw_quals[MAXSAMPLE];

    //more structured data
    int counts[MAXSAMPLE][MAX_IDX];
    double freqs[MAXSAMPLE][MAX_IDX];
    char** ins_bases[MAXSAMPLE];
    char** del_bases[MAXSAMPLE];

    //mutation data
    char mut_type[4];
    char mut_base;
    double mut_score;
    char mut_indel[MAX_INDEL_LEN];
    int mut_sample_idx;
    int * mutated_or_not;
    int cov_in_weakest;
    double mut_freq;
    double cleanliness;

};

/*
    the ploidy regions struct
*/
struct ploidy
{

    //position data
    char* chrom;
    int pos_from;
    int pos_to;

    //ploidy data
    int ploidy_est;

};

////////////////////////////////////////////////////////////////////////////
// initializing, deleting, and copying pileup struct
////////////////////////////////////////////////////////////////////////////

/*
    initializes pointers with NULL
*/
int init_mplp(struct mplp* my_mplp);

/*
    frees all malloced objects in mplp struct,
    and initializes them to NULL
*/
int free_mplp(struct mplp* my_mplp);

/*
    deep copy mpileup line
*/
int copy_mplp(struct mplp* target ,struct mplp* from);


////////////////////////////////////////////////////////////////////////////
// formatted printing of pileup struct
////////////////////////////////////////////////////////////////////////////

/*
    printf formatted representation of mpileup line
*/
int print_mplp(struct mplp* my_mplp);
int print_mplp_basic_info(struct mplp* my_mplp);


////////////////////////////////////////////////////////////////////////////
// process input mpileup line from samtools mpileup command output
////////////////////////////////////////////////////////////////////////////

/*
      process input mpileup line from samtools mpileup command output
*/
int process_mplp_input_line(struct mplp* my_mplp,char* line, ssize_t line_size,
                            int baseq_limit,char** sample_names,int n_sample_names);

////////////////////////////////////////////////////////////////////////////
// read pileup struct from mpileup line
////////////////////////////////////////////////////////////////////////////


/*
    gets next entry from tab separated input string as null terminated char*
*/
int get_next_entry(char* line, ssize_t line_size, ssize_t* pointer, char** result);

/*
    gets mpileup line from the char* line
*/
int get_mplp(struct mplp* my_mplp,char* line, ssize_t read);


////////////////////////////////////////////////////////////////////////////
// Count bases in mplieup struct
////////////////////////////////////////////////////////////////////////////

/*
    counts bases in all samples
*/
int count_bases_all_samples(struct mplp* my_mplp,int baseq_lim);


/*
    counts bases in one sample
*/
int count_bases(char* bases,char* quals,int* counts,char ref_base,int baseq_lim);

/*
    parse a base from the bases and quals
*/
int handle_base(char* bases,char* quals,int* counts, int* filtered_cov,int* base_ptr,int* qual_ptr,int baseq_lim);


/*
    parse a deletion from the bases and quals
*/
int handle_deletion(char* bases,int* del_count,int* base_ptr, char qual,int baseq_lim);

/*
    parse an insertion from the bases and quals
*/
int handle_insertion(char* bases,int* ins_count,int* base_ptr, char qual,int baseq_lim);


////////////////////////////////////////////////////////////////////////////
// Calculate base + ins + del frequencies
////////////////////////////////////////////////////////////////////////////


/*
    calculate base freqs in all samples
*/
int calculate_freqs_all_samples(struct mplp* my_mplp);


/*
   calculate freqs in a sample
*/
int calculate_freqs(double* freqs,int* counts);


////////////////////////////////////////////////////////////////////////////
// Collect indels
////////////////////////////////////////////////////////////////////////////

/*
    collect indels in all samples
*/
int collect_indels_all_samples(struct mplp* my_mplp,int baseq_lim);

/*
    collect the inserted, and deleted bases
*/
int collect_indels(char* bases,char* quals, char*** ins_bases, int ins_count,
                   char*** del_bases,int del_count, int baseq_lim);

/*
    free memory of indel bases
*/
int free_indel_bases(char*** ins_bases,char*** del_bases);



////////////////////////////////////////////////////////////////////////////
// Proximal gap filtering related functions
////////////////////////////////////////////////////////////////////////////


/*
    updates last indel gap position if there is one at the position
*/
int update_last_gap(struct mplp* my_mplp, char** last_gap_chrom,
                    int* last_gap_pos_start, int* last_gap_pos_end, int* is_gap);


/*
    if position is gap delete all potential mutations too close
*/
int proximal_gap_hindsight_filter(struct mplp* potential_mut_lines,int* mut_ptr,
                                  char* last_gap_chrom,int last_gap_pos_start,
                                 int proximal_gap_min_dist_SNV,int proximal_gap_min_dist_indel);

/*
    prints and deletes mutations, which have survived the hindsight proximal gap filtering
*/
int flush_accepted_mutations(struct mplp* potential_mut_lines,
                             char* recent_chrom,int recent_pos,int* mut_ptr,
                             int proximal_gap_min_dist_SNV,int proximal_gap_min_dist_indel);

////////////////////////////////////////////////////////////////////////////
// Call SNV
////////////////////////////////////////////////////////////////////////////

/*
    print mutation
*/
int print_mutation(struct mplp* my_mplp);

/*
    calls SNVs (single nucleotide mutations) from frequencies
*/

int call_snv_with_pl(struct mplp* saved_mutations, int* mut_ptr, struct mplp* my_mplp,
             double sample_mut_freq_limit,double min_other_ref_freq_limit,int cov_limit,
             char* last_gap_chrom, int last_gap_pos_end, int proximal_gap_min_distance,
             int *ploidy_id, int *next_pos, int *current_ploidy, int * max_number_of_ranges, struct ploidy ** pl_r_array, int unique_only,
             int unique_ploidy, int shared_by_all);

/*
    gets the highest not reference mut freq
*/
int get_max_non_ref_freq(struct mplp* my_mplp,double* val, int* idx, char* mut_base);

/*
    checks if there are any noisy samples in the given position,
    selects noisiest clean sample
    and weakest mutated sample
*/

int check_if_noisy(struct mplp* my_mplp, int ref_idx, int alt_idx, double sample_mut_freq_limit,double min_other_ref_freq_limit, int cov_limit,
                  int* current_ploidy, int* noisiest_idx, int* weakest_idx, int shared_by_all);

/*
    gets the lowest reference freq, except for 1 sample
*/
int get_min_ref_freq(struct mplp* my_mplp,double* val, int idx_2skip, int* other_idx );

/*
    Fisher's exact
*/
double fisher22(uint32_t m11, uint32_t m12, uint32_t m21, uint32_t m22, uint32_t midp);

////////////////////////////////////////////////////////////////////////////
// Call indel
////////////////////////////////////////////////////////////////////////////

/*
    calls indels
*/

int call_indel_with_pl(struct mplp* saved_mutations, int* mut_ptr, struct mplp* my_mplp,
               double sample_mut_freq_limit,double min_other_ref_freq_limit,int cov_limit,
               char* last_gap_chrom, int last_gap_pos_start,int last_gap_pos_end,
               int prox_gap_min_dist_SNV,int prox_gap_min_dist_indel,
               int *ploidy_id, int *next_pos, int *current_ploidy, int *max_number_of_ranges, struct ploidy ** pl_r_array, int unique_only,
             int unique_ploidy, int shared_by_all);


/*
    gets the highest indel freq
*/
int get_max_indel_freq(struct mplp* my_mplp,double* val, int* idx, char* mut_indel,char* mut_type);

/*
    gets the indel supported by the largest number of reads from all samples combined
*/
int get_most_frequent_indel(struct mplp* my_mplp,char* most_frequent_indel,char* mut_type);

/*
    checks if there are any noisy samples in the given position (for indels),
    determines coverage and ref_freq for the noisiest clean sample,
    and coverage and mut_freq for the weakest mutated sample
*/
int check_if_indel_noisy(struct mplp* my_mplp, double sample_mut_freq_limit, double min_other_ref_freq_limit, int cov_limit,
                                int* current_ploidy, double* noisiest_reffreq, int* noisiest_cov, double* weakest_mutfreq, int* weakest_cov, int* weakest_id,
                                char* most_frequent_indel, char* mut_type, int shared_by_all);

/*
    gets the highest indel freq, except for 1 sample
*/
int get_min_other_noindel_freq(struct mplp* my_mplp,double* val, int idx_2skip, int* other_idx );

////////////////////////////////////////////////////////////////////////////
// functions to read ploidy information, and manage current ploidy values
////////////////////////////////////////////////////////////////////////////

/*
    number of sample names in string (comma-separated)
*/
int get_token_num (char* sample_string);

/*
    string to array (comma-separated)
*/
int fill_array (char * pointer_to_array[], char* sample_string);

/*
    basic ploidy information
*/
int get_basic_ploidy_info(char* ploidy_info_filename, int * number_of_ploidy_files, int * max_number_of_ranges);

/*
    fill array with ploidy information
*/
int build_ploidy_ranges_array(char* ploidy_info_filename, struct ploidy ** pl_r_array, int * ploidy_id, int * next_pos_to_check, char** sample_names, int * total_sample_num);

/*
    update ploidy for single sample
*/
int update_ploidy(struct mplp* my_mplp, int *sample_id, int *ploidy_id, int *next_pos, int *current_ploidy, int *max_number_of_ranges, struct ploidy ** pl_r_array, int default_ploidy);

/*
    update ploidy for all samples
*/
int update_ploidy_all(struct mplp* my_mplp, int *ploidy_id, int *next_pos, int *current_ploidy, int *max_number_of_ranges, struct ploidy ** pl_r_array, int default_ploidy);

/*
    checks for ploidy file
*/
int check_for_ploidy_file(char* ploidy_filename);

/*
    initializes ploidy structure
*/
int init_ploidy(struct ploidy* my_ploidy);

/*
    frees ploidy structure
*/
int free_ploidy(struct ploidy* my_ploidy);

/*
    reads ploidy data from external file
*/
int load_ploidy_file_to_array(char* ploidy_filename, struct ploidy * ploidy_array);

/*
    looks up ploidy index for current chrom and pos
*/
int find_ploidy_idx(struct ploidy * ploidy_array,int n, char *ch, int pos);

////////////////////////////////////////////////////////////////////////////
// preparing temp files for ploidy estimation
////////////////////////////////////////////////////////////////////////////

/*
    checks if the position is interesting enough
*/

int process_position_for_ploidy_est(struct mplp* my_mplp,
             double min_noise,int min_cov,int max_cov,
             int* dip_cov_total, int* dip_count, int* trip_cov_total, int* trip_count);

/*
    store relevant mpileup data for a window of a given size
*/

int build_window_data(double ** window_data, char ** ch, int * pointer_wd, struct mplp* my_mplp);

/*
    prints the beginning of the window and shifts data up
*/

int shift_window(double ** window_data, char ** ch, int * pointer_wd, int ws, int rows, int shift,
             double min_noise, int print_every_nth);

/*
    prints the last non-filled window to file
*/

int print_last_window(double ** window_data, char ** ch, int ws,
             double min_noise, int print_every_nth);
