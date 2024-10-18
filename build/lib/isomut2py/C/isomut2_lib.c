#include "isomut2_lib.h"


////////////////////////////////////////////////////////////////////////////
// initialize
////////////////////////////////////////////////////////////////////////////
/*
    initializes all the pointers with NULL
        - always call this first after creating the struct,
            to avoid ERROR due to freeing/using memory garbage
*/
int init_mplp(struct mplp* my_mplp){
    int i;
    my_mplp->raw_line=NULL;
    my_mplp->chrom=NULL;
    my_mplp->mutated_or_not = NULL; //new
    my_mplp->pos=-42;
    my_mplp->n_samples=-42;
    my_mplp->ref_nuq='E';

    for(i=0;i<MAXSAMPLE;i++){
        my_mplp->raw_bases[i]=NULL;
        my_mplp->raw_quals[i]=NULL;
        my_mplp->ins_bases[i]=NULL;
        my_mplp->del_bases[i]=NULL;
    }


    strcpy(my_mplp->mut_type,"NOT\0");
    my_mplp->mut_base='E';
    my_mplp->mut_score=-42;
    for (i=0;i<MAX_INDEL_LEN;i++){my_mplp->mut_indel[i]='.';}
    my_mplp->mut_sample_idx=-42;
    my_mplp->mut_freq=-42;
    my_mplp->cov_in_weakest=-42;
    my_mplp->cleanliness=-42;
    my_mplp->ploidy=-42;

    return 0;
}

////////////////////////////////////////////////////////////////////////////
// free resources
////////////////////////////////////////////////////////////////////////////
/*
    frees all malloced objects in mplp struct
        - dont try to free uninitialized struct!
*/
int free_mplp(struct mplp* my_mplp){
    if( my_mplp->raw_line !=NULL) free(my_mplp->raw_line);
    if( my_mplp->chrom !=NULL) free(my_mplp->chrom);
    if (my_mplp->mutated_or_not != NULL) free(my_mplp->mutated_or_not);

    int i,j;
    for(i=0;i<my_mplp->n_samples;i++){
        if( my_mplp->raw_bases[i] !=NULL) free(my_mplp->raw_bases[i]);
        if( my_mplp->raw_quals[i] !=NULL) free(my_mplp->raw_quals[i]);

        if( my_mplp->ins_bases[i] !=NULL){
            for(j=0;j<my_mplp->counts[i][INS_START_IDX];j++){
                free(my_mplp->ins_bases[i][j]);
            }
            free(my_mplp->ins_bases[i]);
        }
        if( my_mplp->del_bases[i] !=NULL){
            for(j=0;j<my_mplp->counts[i][DEL_START_IDX];j++){
                free(my_mplp->del_bases[i][j]);
            }
            free(my_mplp->del_bases[i]);
        }
    }
    //initalize the pointers to null after freeing
    init_mplp(my_mplp);
    return 0;
}


////////////////////////////////////////////////////////////////////////////
// deep copy pileup struct
////////////////////////////////////////////////////////////////////////////

/*
    deep copy mpileup line
*/
int copy_mplp(struct mplp* target ,struct mplp* from) {
    //free and initialize resources int the target
    free_mplp(target);

    target->sample_names=from->sample_names;
    target->n_sample_names=from->n_sample_names;

    //copy raw line and chromosome
    target->raw_line = (char*) malloc((strlen(from->raw_line)+1) * sizeof(char));
    strcpy( target->raw_line, from->raw_line);
    target->chrom = (char*) malloc((strlen(from->chrom)+1) * sizeof(char));
    strcpy( target->chrom, from->chrom);
    //copy the position level data
    target->pos=from->pos;
    target->ref_nuq=from->ref_nuq;
    target->n_samples=from->n_samples;
    ///copy the mutation related data
    target->mut_base=from->mut_base;
    target->mut_sample_idx=from->mut_sample_idx;
    target->mut_score=from->mut_score;
    target->mut_freq=from->mut_freq;
    target->cov_in_weakest=from->cov_in_weakest;
    target->cleanliness=from->cleanliness;
    target->ploidy=from->ploidy;
    strcpy( target->mut_type, from->mut_type);
    strncpy( target->mut_indel, from->mut_indel,MAX_INDEL_LEN);
    //loop over sample level data
    int i,j;
    target->mutated_or_not=malloc(target->n_samples * sizeof(int));
    for(i=0;i<target->n_samples;i++){
        //copy basic data cov, bases, quals
        target->raw_cov[i]=from->raw_cov[i];
        target->mutated_or_not[i]=from->mutated_or_not[i];
        target->raw_bases[i] = (char*) malloc((strlen(from->raw_bases[i])+1) * sizeof(char));
        strcpy( target->raw_bases[i], from->raw_bases[i]);
        target->raw_quals[i] = (char*) malloc((strlen(from->raw_quals[i])+1) * sizeof(char));
        strcpy( target->raw_quals[i], from->raw_quals[i]);
        //copy filtered data, counts, coverage
        for(j=0;j<MAX_IDX;j++){
            target->counts[i][j]=from->counts[i][j];
            target->freqs[i][j]=from->freqs[i][j];
        }
        //copy all indel sequences
        target->ins_bases[i] = (char**) malloc(target->counts[i][INS_START_IDX] * sizeof(char*));
        target->del_bases[i] = (char**) malloc(target->counts[i][DEL_START_IDX] * sizeof(char*));
        for (j=0;j<target->counts[i][INS_START_IDX];j++){
            target->ins_bases[i][j] = (char*) malloc((strlen(from->ins_bases[i][j])+1) * sizeof(char));
            strcpy( target->ins_bases[i][j], from->ins_bases[i][j]);
        }
        for (j=0;j<target->counts[i][DEL_START_IDX];j++){
            target->del_bases[i][j] = (char*) malloc((strlen(from->del_bases[i][j])+1) * sizeof(char));
            strcpy( target->del_bases[i][j], from->del_bases[i][j]);
        }
    }
    return 0;
}


////////////////////////////////////////////////////////////////////////////
// formatted printing functions
////////////////////////////////////////////////////////////////////////////


/*
    printf formatted representation of mpileup line
*/
int print_mplp(struct mplp* my_mplp){
    //print position level info
    printf("%s %d %c\n",my_mplp->chrom,my_mplp->pos,my_mplp->ref_nuq);

    //print counts and freqs and bases and quals
    int i;
    for(i=0;i<my_mplp->n_samples;i++){
        printf("cov %d,A %d,C %d,G %d,T %d,del %d,ins_start %d, del_start %d,read_start %d, read_end %d\n",
               my_mplp->counts[i][COV_IDX],
               my_mplp->counts[i][A_IDX],
               my_mplp->counts[i][C_IDX],
               my_mplp->counts[i][G_IDX],
               my_mplp->counts[i][T_IDX],
               my_mplp->counts[i][DEL_IDX],
               my_mplp->counts[i][INS_START_IDX],
               my_mplp->counts[i][DEL_START_IDX],
               my_mplp->counts[i][READ_START_IDX],
               my_mplp->counts[i][READ_END_IDX]);
        printf("A %.2f,C %.2f,G %.2f,T %.2f,del %.2f,ins_start %.2f, del_start %.2f,read_start %.2f, read_end %.2f\n",
               my_mplp->freqs[i][A_IDX],
               my_mplp->freqs[i][C_IDX],
               my_mplp->freqs[i][G_IDX],
               my_mplp->freqs[i][T_IDX],
               my_mplp->freqs[i][DEL_IDX],
               my_mplp->freqs[i][INS_START_IDX],
               my_mplp->freqs[i][DEL_START_IDX],
               my_mplp->freqs[i][READ_START_IDX],
               my_mplp->freqs[i][READ_END_IDX]);
        printf("%d %s %s\n",my_mplp->raw_cov[i],my_mplp->raw_bases[i],my_mplp->raw_quals[i]);
    }
    return 0;
}

int print_mplp_basic_info(struct mplp* my_mplp){
    //print position level info
    printf("%s\t%d\t%c\t",my_mplp->chrom,my_mplp->pos,my_mplp->ref_nuq);

    int i;
    for(i=0;i<my_mplp->n_samples;i++){
        if (i<my_mplp->n_samples-1){
            printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t",
                   my_mplp->counts[i][COV_IDX],
                   my_mplp->freqs[i][A_IDX],
                   my_mplp->freqs[i][C_IDX],
                   my_mplp->freqs[i][G_IDX],
                   my_mplp->freqs[i][T_IDX],
                   my_mplp->freqs[i][INS_START_IDX],
                   my_mplp->freqs[i][DEL_START_IDX]);
        }
        else{
            printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
                   my_mplp->counts[i][COV_IDX],
                   my_mplp->freqs[i][A_IDX],
                   my_mplp->freqs[i][C_IDX],
                   my_mplp->freqs[i][G_IDX],
                   my_mplp->freqs[i][T_IDX],
                   my_mplp->freqs[i][INS_START_IDX],
                   my_mplp->freqs[i][DEL_START_IDX]);
        }
    }
    return 0;
}
////////////////////////////////////////////////////////////////////////////
// process input mpileup line from samtools mpileup command output
////////////////////////////////////////////////////////////////////////////

/*
      process input mpileup line from samtools mpileup command output
*/
int process_mplp_input_line(struct mplp* my_mplp,char* line, ssize_t line_size,
                            int baseq_limit,char** sample_names,int n_sample_names){

    //read mpileup
    get_mplp(my_mplp,line,line_size);

    //assert if n_sampls==n_sample_names
    if(n_sample_names == my_mplp->n_samples){
        my_mplp->n_sample_names=n_sample_names;
        my_mplp->sample_names=sample_names;
    }
    else{
        printf("ERROR, length of sample names != number of samples in input stream\n ");
        exit(1);
    }

    //count bases
    count_bases_all_samples(my_mplp,baseq_limit);

    //calculate freqs
    calculate_freqs_all_samples(my_mplp);

    //collect indels
    collect_indels_all_samples(my_mplp,baseq_limit);

    return 0;
}



////////////////////////////////////////////////////////////////////////////
// read pileup struct from mpileup line
////////////////////////////////////////////////////////////////////////////

/*
    gets mpileup line from the char* line
*/
int get_mplp(struct mplp* my_mplp,char* line, ssize_t line_size){
    //store the raw line too
    free(my_mplp->raw_line);
    my_mplp->raw_line = (char*)malloc( (line_size+1) * sizeof(char));
    strcpy(my_mplp->raw_line,line);

    //temp buffer for reading those entries which will be formatted as not strings
    char* tmp_str=NULL;

    ssize_t i=0;
    while(i<line_size){
        //chrom
        get_next_entry(line,line_size,&i,&(my_mplp->chrom));
        //position
        get_next_entry(line,line_size,&i,&tmp_str);
        my_mplp->pos= (int) strtol(tmp_str,NULL,10);
        //ref nuq
        get_next_entry(line,line_size,&i,&tmp_str);
        my_mplp->ref_nuq=(char) toupper(tmp_str[0]);

        //read samples
        int temp_sample=0;
        while(i<line_size){
            //coverage
            get_next_entry(line,line_size,&i,&tmp_str);
            my_mplp->raw_cov[temp_sample]= (int) strtol(tmp_str,NULL,10);
            //bases
            get_next_entry(line,line_size,&i,&(my_mplp->raw_bases[temp_sample]));
            //quals
            get_next_entry(line,line_size,&i,&(my_mplp->raw_quals[temp_sample]));
            temp_sample++;
        }
        //store the number of samples
        my_mplp->n_samples=temp_sample;
    }
    //free resources
    free(tmp_str);
    return 0;
}

/*
    gets next entry from tab separated input string as null terminated char*
*/
int get_next_entry(char* line, ssize_t line_size, ssize_t* pointer, char** result){
    int c,c0;
    c0 = (int) *pointer;
    while(line[*pointer]!='\t' && *pointer<line_size)(*pointer)++;
    c = (int) *pointer-c0;
    (*pointer)++;

    if(*result != NULL) free(*result);
    *result = (char*)malloc( (c+1) * sizeof(char));
    memcpy(*result,line+c0,c * sizeof(char));
    (*result)[c] = 0;

    return c;
}



////////////////////////////////////////////////////////////////////////////
// Count bases in mplieup struct
////////////////////////////////////////////////////////////////////////////

/*
    counts bases,indels, and read start end signs in all samples
*/
int count_bases_all_samples(struct mplp* my_mplp,int baseq_lim){
    int i=0;
    for(i=0;i<my_mplp->n_samples;i++){
        count_bases(my_mplp->raw_bases[i],my_mplp->raw_quals[i],my_mplp->counts[i],
                    my_mplp->ref_nuq,baseq_lim);
    }
    return 0;
}

/*
    counts bases in one sample
*/
int count_bases(char* bases, char* quals,int* counts,char ref_base,int baseq_lim){
    //initialize counts to zero
    int i,j;
    for( i=0;i<MAX_IDX;i++) counts[i]=0;

    i = 0; //pointer in str for bases
    j = 0; //pointer in str for qualities
    counts[COV_IDX]=0;
    while(bases[i]!=0){
        //beginning and end of the read signs
        if(bases[i] == '^' ) {i+=2;counts[READ_START_IDX]++;}
        else if(bases[i] == '$' ) {i++,counts[READ_END_IDX]++;}
        //deletions
        else if(bases[i]=='-' ) handle_deletion(bases,&counts[DEL_START_IDX],&i,quals[j-1],baseq_lim);
        //insetions
        else if(bases[i]=='+' ) handle_insertion(bases,&counts[INS_START_IDX],&i,quals[j-1],baseq_lim);
        //real base data
        else handle_base(bases,quals,counts,&counts[COV_IDX],&i,&j,baseq_lim);
    }
    //add refbase to corresponding base
    if(ref_base=='A') counts[A_IDX]+=counts[REF_IDX];
    else if(ref_base=='C') counts[C_IDX]+=counts[REF_IDX];
    else if(ref_base=='G') counts[G_IDX]+=counts[REF_IDX];
    else if(ref_base=='T') counts[T_IDX]+=counts[REF_IDX];
    return 0;
}


/*
    parse a base from the bases and quals
*/
int handle_base(char* bases,char* quals,int* base_counts, int* filtered_cov,
                   int* base_ptr,int* qual_ptr,int baseq_lim){

    char c = bases[*base_ptr];
    if(quals[*qual_ptr] >= baseq_lim + 33 ){
        if(c=='.' || c==',' )      base_counts[REF_IDX]++;
        else if(c=='A' || c=='a' ) base_counts[A_IDX]++;
        else if(c=='C' || c=='c' ) base_counts[C_IDX]++;
        else if(c=='G' || c=='g' ) base_counts[G_IDX]++;
        else if(c=='T' || c=='t' ) base_counts[T_IDX]++;
        else if(c=='*' ) base_counts[DEL_IDX]++;
        (*filtered_cov)++;
    }
    (*qual_ptr)++;
    (*base_ptr)++;

    return 0;
}

/*
    parse a deletion from the bases and quals
*/
int handle_deletion(char* bases,int* del_count,int* base_ptr,char qual,int baseq_lim){
    char* offset;
    int indel_len= (int) strtol(&bases[*base_ptr+1],&offset,10);
    (*base_ptr)+= offset-&bases[*base_ptr] + indel_len;
    if(qual >= baseq_lim + 33 )(*del_count)++;
    return 0;
}

/*
    parse an insertion from the bases and quals
*/
int handle_insertion(char* bases,int* ins_count,int* base_ptr,char qual,int baseq_lim){
    char* offset;
    int indel_len= (int) strtol(&bases[*base_ptr+1],&offset,10);
    (*base_ptr)+= offset-&bases[*base_ptr] + indel_len;
    if(qual >= baseq_lim + 33 ) (*ins_count)++;
    return 0;
}


///////////////////////////////////////////////////////////////////////////
// Calculate base frequencies
////////////////////////////////////////////////////////////////////////////

/*
    calculate base freqs in all samples
*/
int calculate_freqs_all_samples(struct mplp* my_mplp){
    int i=0;
    for(i=0;i<my_mplp->n_samples;i++){
        calculate_freqs(my_mplp->freqs[i],my_mplp->counts[i]);
    }
    return 0;
}


/*
   calculate freqs in a sample
*/
int calculate_freqs(double* freqs,int* counts){
    int i;
    if (counts[COV_IDX]!=0){
        for(i=0;i<MAX_IDX;i++){
            freqs[i]=( (double) counts[i]) / counts[COV_IDX];
        }
    }
    else {
        for(i=0;i<MAX_IDX;i++){
            freqs[i]= ZERO_COV_FREQ;
        }
    }
    return 0;
}


////////////////////////////////////////////////////////////////////////////
// Collect indels
////////////////////////////////////////////////////////////////////////////

/*
    collect indels in all samples
*/
int collect_indels_all_samples(struct mplp* my_mplp,int baseq_lim){
    int i;
    for(i=0;i<my_mplp->n_samples;i++){
        collect_indels(my_mplp->raw_bases[i],
                       my_mplp->raw_quals[i],
                       &my_mplp->ins_bases[i],
                       my_mplp->counts[i][INS_START_IDX],
                       &my_mplp->del_bases[i],
                       my_mplp->counts[i][DEL_START_IDX],
                       baseq_lim);
    }
    return 0;
}

/*
    collect the inserted, and deleted bases
*/
int collect_indels(char* bases,char* quals, char*** ins_bases, int ins_count,
                   char*** del_bases,int del_count, int baseq_lim){
    //allocate new memory
    if( *ins_bases!=NULL || *del_bases!=NULL){
        printf("ERROR: collect_indels() called on not NULL ins_bases, del_bases pointers,\n");
        printf("       maybe its called 2nd time, or mplp struct not freed, or not initialized,\n");
        printf("       possible memory leak, exiting");
        exit(1);
    }
    *ins_bases = (char**) malloc( (ins_count) * sizeof(char*));
    *del_bases = (char**) malloc( (del_count) * sizeof(char*));

    int i,j,del_c,ins_c; //pointers in data
    i = j = del_c = ins_c = 0;
    char* offset;
    while(bases[i]!=0){
        //beginning and end of the read signs
        if(bases[i] == '$' ) i++; //next
        else if(bases[i] == '^' ) i+=2; //jump next character (mapq too)
        //deletions
        else if(bases[i]=='-' ) {
            int indel_len= (int) strtol(&bases[i+1],&offset,10);
            i+= offset-&bases[i] + indel_len;
            if( quals[j-1] >= baseq_lim + 33 ){
                (*del_bases)[del_c] = (char*) malloc( (indel_len+1) * sizeof(char));
                memcpy((*del_bases)[del_c],offset,indel_len * sizeof(char));
                (*del_bases)[del_c][indel_len]=0;
                del_c++;
            }
        }
        //insertions
        else if(bases[i]=='+' ){
            int indel_len= (int) strtol(&bases[i+1],&offset,10);
            i+= offset-&bases[i] + indel_len;
            if( quals[j-1] >= baseq_lim + 33 ){
                (*ins_bases)[ins_c] = (char*) malloc( (indel_len+1) * sizeof(char));
                memcpy((*ins_bases)[ins_c],offset,indel_len * sizeof(char));
                (*ins_bases)[ins_c][indel_len]=0;
                ins_c++;
            }
        }
        //real base data
        else {i++;j++;}
    }
    return 0;
}


////////////////////////////////////////////////////////////////////////////
// Proximal gap filtering
////////////////////////////////////////////////////////////////////////////

/*
    updates last indel gap position if there is one at the position
*/
int update_last_gap(struct mplp* my_mplp, char** last_gap_chrom,
                    int* last_gap_pos_start, int* last_gap_pos_end, int* is_gap){
    *is_gap=1;
    int i,j;
    for(i=0;i<my_mplp->n_samples;i++){
        //last gap is either an insertion start deletion start or 0 cov
        // or more than 50% read start, read end
        if (my_mplp->counts[i][INS_START_IDX]!=0 ||
            my_mplp->counts[i][DEL_START_IDX]!=0 ||
            my_mplp->freqs[i][READ_START_IDX] > 0.5 ||
            my_mplp->freqs[i][READ_END_IDX] > 0.5 ){

            //copy last chrom
            if ( *last_gap_chrom != NULL ) free(*last_gap_chrom);
            *last_gap_chrom = (char*) malloc( (strlen(my_mplp->chrom)+1) * sizeof(char));
            strcpy(*last_gap_chrom,my_mplp->chrom);
            //update last gap pos start
            *last_gap_pos_start = my_mplp->pos;
            //update last gap pos end for 0 coverage, read starts, ends
            if ( *last_gap_pos_start > *last_gap_pos_end ){
                 *last_gap_pos_end = *last_gap_pos_start;
             }
            //update last gap pos end for indels
             if ( *last_gap_pos_start >= *last_gap_pos_end  && my_mplp->counts[i][INS_START_IDX]!=0 ){
                 *last_gap_pos_end = *last_gap_pos_start + 1;
             }
            //gap pos end is hihger for deletions
            for (j=0;j < my_mplp->counts[i][DEL_START_IDX];j++){
                if(*last_gap_pos_end < my_mplp->pos + 1 + (int) strlen(my_mplp->del_bases[i][j])){
                    *last_gap_pos_end = my_mplp->pos + 1 + (int) strlen(my_mplp->del_bases[i][j]);
                }
            }
            //set inidicator
            *is_gap=0;
            break;
        }
    }
    return 0;
}

/*
    if position is gap delete all potential mutations too close
*/
int proximal_gap_hindsight_filter(struct mplp* saved_mutations,int* mut_ptr,
                                  char* last_gap_chrom,int last_gap_pos_start,
                                 int proximal_gap_min_dist_SNV,int proximal_gap_min_dist_indel){

    //if position is a called indel the filter has been run
    if( *mut_ptr > 0 &&  strcmp(last_gap_chrom,saved_mutations[(*mut_ptr) -1].chrom) == 0 &&
            last_gap_pos_start == saved_mutations[(*mut_ptr) -1].pos ){
        return 0;
    }
    int i,j;
    for(i= *mut_ptr-1;i>=0;i--){
        //delet SNV too close to gap
        if (strcmp(last_gap_chrom,saved_mutations[i].chrom) != 0) return 0;
        else if( strcmp(saved_mutations[i].mut_type,"SNV")==0 &&
                    last_gap_pos_start - saved_mutations[i].pos  < proximal_gap_min_dist_SNV ){
            //delete this line
            free_mplp(&saved_mutations[i]);
            (*mut_ptr)--;
        }
        //delete indels too close to gap and quit, because filter was run then
        else if( ( strcmp(saved_mutations[i].mut_type,"INS")==0 && //ins
                   last_gap_pos_start - saved_mutations[i].pos  < proximal_gap_min_dist_indel)
                   || ( strcmp(saved_mutations[i].mut_type,"DEL")==0  &&  //del
                   last_gap_pos_start - saved_mutations[i].pos -
                   ((int)strlen(saved_mutations[i].mut_indel))  < proximal_gap_min_dist_indel) ){
            //delete this line
            free_mplp(&saved_mutations[i]);
            (*mut_ptr)--;
            //copy all later SNVs one position ahead
            for(j=i;j < *mut_ptr;j++){
                copy_mplp(&saved_mutations[j],&saved_mutations[j+1]);
                free_mplp(&saved_mutations[j+1]);
            }
            return 0;
        }
    }
    return 0;

}

/*
    prints and deletes mutations, which have survived the hindsight proximal gap filtering
*/
int flush_accepted_mutations(struct mplp* saved_mutations,
                             char* recent_chrom,int recent_pos,int* mut_ptr,
                             int proximal_gap_min_dist_SNV,int proximal_gap_min_dist_indel){
    int i,j;
    for(i = 0; i<*mut_ptr;i++){
        //check if mut is accepted already
        if ( ( strcmp(recent_chrom,saved_mutations[i].chrom) != 0 ) || //other chrom
             ( strcmp(saved_mutations[i].mut_type,"SNV")==0 && //SNV
             recent_pos - saved_mutations[i].pos  > proximal_gap_min_dist_SNV ) ||
             ((strcmp(saved_mutations[i].mut_type,"INS")==0 || //indel
               strcmp(saved_mutations[i].mut_type,"DEL")==0 ) &&
             recent_pos - saved_mutations[i].pos  > proximal_gap_min_dist_indel )){
            //print and delete it
            print_mutation(&saved_mutations[i]);
            free_mplp(&saved_mutations[i]);
            (*mut_ptr)--;
            //step everyone else ahead
            for(j=i;j < *mut_ptr;j++){
                copy_mplp(&saved_mutations[j],&saved_mutations[j+1]);
                free_mplp(&saved_mutations[j+1]);
            }
            i--;
        }
    }
    return 0;
}


////////////////////////////////////////////////////////////////////////////
// Call Mutations
////////////////////////////////////////////////////////////////////////////

/*
    print mutation
*/
int print_mutation(struct mplp* my_mplp){
    int i;
    int number_of_printed_samples = 0;
    if ( strcmp(my_mplp->mut_type,"SNV") ==0 ){
        //printing all samples where the mutation is found
        for (i=0;i<my_mplp->n_samples;i++)
        {
          if (my_mplp->mutated_or_not[i] == 1)
          {
            if (number_of_printed_samples != 0){printf(",%s", my_mplp->sample_names[i]);}
            else{printf("%s", my_mplp->sample_names[i]); number_of_printed_samples++;}
          }
        }
        printf("\t");
        printf("%s\t%d\t%s\t%.2f\t%c\t%c\t%d\t%.3f\t%.3f\t%d\n",
               my_mplp->chrom,
               my_mplp->pos,
               my_mplp->mut_type,
               my_mplp->mut_score,
               my_mplp->ref_nuq,
               my_mplp->mut_base,
               my_mplp->cov_in_weakest,
               my_mplp->mut_freq,
               my_mplp->cleanliness,
               my_mplp->ploidy);
    }
    if ( strcmp(my_mplp->mut_type,"INS") ==0   ){
        //printing all samples where the mutation is found
        for (i=0;i<my_mplp->n_samples;i++)
        {
          if (my_mplp->mutated_or_not[i] == 1)
          {
            if (number_of_printed_samples != 0){printf(",%s", my_mplp->sample_names[i]);}
            else{printf("%s", my_mplp->sample_names[i]); number_of_printed_samples++;}
          }
        }
        printf("\t");
        printf("%s\t%d\t%s\t%.2f\t-\t%s\t%d\t%.3f\t%.3f\t%d\n",
               my_mplp->chrom,
               my_mplp->pos,
               my_mplp->mut_type,
               my_mplp->mut_score,
               my_mplp->mut_indel,
               my_mplp->cov_in_weakest,
               my_mplp->mut_freq,
               my_mplp->cleanliness,
               my_mplp->ploidy);

    }
    if ( strcmp(my_mplp->mut_type,"DEL") ==0   ){
        //printing all samples where the mutation is found
        for (i=0;i<my_mplp->n_samples;i++)
        {
          if (my_mplp->mutated_or_not[i] == 1)
          {
            if (number_of_printed_samples != 0){printf(",%s", my_mplp->sample_names[i]);}
            else{printf("%s", my_mplp->sample_names[i]); number_of_printed_samples++;}
          }
        }
        printf("\t");
        printf("%s\t%d\t%s\t%.2f\t%s\t-\t%d\t%.3f\t%.3f\t%d\n",
               my_mplp->chrom,
               my_mplp->pos,
               my_mplp->mut_type,
               my_mplp->mut_score,
               my_mplp->mut_indel,
               my_mplp->cov_in_weakest,
               my_mplp->mut_freq,
               my_mplp->cleanliness,
               my_mplp->ploidy);

    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////
// Call SNVs
////////////////////////////////////////////////////////////////////////////


/*
    calls mutation from frequencies
*/

int call_snv_with_pl(struct mplp* saved_mutations, int* mut_ptr, struct mplp* my_mplp,
             double sample_mut_freq_limit,double min_other_ref_freq_limit,int cov_limit,
             char* last_gap_chrom, int last_gap_pos_end, int proximal_gap_min_distance,
             int *ploidy_id, int *next_pos, int *current_ploidy, int * max_number_of_ranges, struct ploidy ** pl_r_array,
              int unique_only, int default_ploidy, int shared_by_all){

     //filter position if it is too close to last gap
    if ( last_gap_chrom != NULL && //no gap yet
         strcmp(last_gap_chrom,my_mplp->chrom) == 0 && //same chrom
         my_mplp->pos - last_gap_pos_end < proximal_gap_min_distance ){ // proximal gap
        return 0;
    }

    //skip position if the reference base is N
    if (my_mplp->ref_nuq == 'N'){
      return 0;
    }

    double sample_mut_freq;
    int sample_idx;
    int noisiest_idx, weakest_idx;
    char mut_base = 'E';

    //get the "most mutated" sample and the its mut_freq
    get_max_non_ref_freq(my_mplp,&sample_mut_freq,&sample_idx,&mut_base);

    //if it's a very clean position for all samples, skip it from the start
    if (sample_mut_freq < 0.02){
      return 0;
    }

    //if we are only looking for unique mutations, it's faster
    if ((unique_only == 1 && sample_idx >=0) || (my_mplp->n_samples == 1)){
        //updating ploidy of the "most mutated" sample
        update_ploidy(my_mplp, &sample_idx, ploidy_id, next_pos, current_ploidy, max_number_of_ranges, pl_r_array, default_ploidy);
        int cp = current_ploidy[sample_idx];
        int other_idx;
        double min_other_ref_freq;
        //getting ref_freq for the noisiest other sample
        int status_mrf = get_min_ref_freq(my_mplp,&min_other_ref_freq,sample_idx,&other_idx);
        if (status_mrf == 0 && // not bad position
          sample_mut_freq >= (sample_mut_freq_limit)*2/cp && // mut_freq higher than limit, accounting for ploidy as well
          min_other_ref_freq > min_other_ref_freq_limit && //other sample ref_freq higher than limit
          my_mplp->counts[sample_idx][COV_IDX] >= cov_limit ){ //coverage higher than limit

          int i;
          //update mutation information in mplp structure
          my_mplp->mut_base=mut_base;
          my_mplp->mutated_or_not = malloc(my_mplp->n_samples * sizeof(int));
          for (i=0;i<my_mplp->n_samples;i++){my_mplp->mutated_or_not[i]=0;}
          my_mplp->mutated_or_not[sample_idx]=1;
          my_mplp->ploidy = cp;
          my_mplp->mut_freq=sample_mut_freq;
          my_mplp->cleanliness=min_other_ref_freq;
          if (my_mplp->n_samples == 1){
            my_mplp->cleanliness=min_other_ref_freq_limit;
          }
          my_mplp->cov_in_weakest=my_mplp->counts[sample_idx][COV_IDX];
          strncpy(my_mplp->mut_type, "SNV\0",4);
          my_mplp->mut_score = -log10(fisher22((uint32_t) ((1-sample_mut_freq) * my_mplp->counts[sample_idx][COV_IDX]),
                                 (uint32_t) (sample_mut_freq * my_mplp->counts[sample_idx][COV_IDX]),
                                 (uint32_t) (my_mplp->cleanliness * my_mplp->counts[other_idx][COV_IDX]),
                                 (uint32_t) ((1-my_mplp->cleanliness) * my_mplp->counts[other_idx][COV_IDX]),1));

          //save potential mutation
          copy_mplp(&saved_mutations[*mut_ptr],my_mplp);
          (*mut_ptr)++;
          return 1;
      }
      return 0;
    }

    //if we are looking for non-unique mutations as well, it's more complicated

    //identifying idx for the reference and the alternate bases for later use
    int base_2_idx[256];
    base_2_idx[ 'A' ] = A_IDX;base_2_idx[ 'C' ] = C_IDX;
    base_2_idx[ 'G' ] = G_IDX;base_2_idx[ 'T' ] = T_IDX;
    int ref_idx=base_2_idx[(int)my_mplp->ref_nuq];
    int alt_idx=base_2_idx[(int)mut_base];

    //updating ploidy values for ALL of the samples
    update_ploidy_all(my_mplp, ploidy_id, next_pos, current_ploidy, max_number_of_ranges, pl_r_array, default_ploidy);

    //if the "most mutated sample" is not mutated enough (using the current ploidy), skip the position
    if (sample_mut_freq < (sample_mut_freq_limit)*2/current_ploidy[sample_idx]){
      return 0;
    }

    //checking if we find a noisy sample at the given position (noisy: mut_freq < sample_mut_freq_limit && ref_freq < min_other_ref_freq_limit && cov >= 0)
    int is_it_noisy = check_if_noisy(my_mplp, ref_idx, alt_idx, sample_mut_freq_limit, min_other_ref_freq_limit, cov_limit,
                                    current_ploidy, &noisiest_idx, &weakest_idx, shared_by_all);
    //if the position has at least one noisy sample, skip it
    if (is_it_noisy != 0){
      return 1; // making sure we don't look for indels if any of the samples is noisy
    }

    //otherwise update mutation information in mplp structure
    my_mplp->mut_base=mut_base;
    my_mplp->ploidy = current_ploidy[weakest_idx];
    my_mplp->mut_freq=my_mplp->freqs[weakest_idx][alt_idx];
    my_mplp->cov_in_weakest=my_mplp->counts[weakest_idx][COV_IDX];
    strncpy(my_mplp->mut_type, "SNV\0",4);
    if (noisiest_idx != -999){
      my_mplp->cleanliness=my_mplp->freqs[noisiest_idx][ref_idx];
      my_mplp->mut_score = -log10(fisher22((uint32_t) ((1-my_mplp->mut_freq) * my_mplp->cov_in_weakest),
                             (uint32_t) (my_mplp->mut_freq * my_mplp->cov_in_weakest),
                             (uint32_t) (my_mplp->cleanliness * my_mplp->counts[noisiest_idx][COV_IDX]),
                             (uint32_t) ((1-my_mplp->cleanliness) * my_mplp->counts[noisiest_idx][COV_IDX]),1));

    }

    //if we did not find any clean samples in the position (meaning that all covered samples are mutated),
    //we use the min_other_ref_freq_limit to estimate the ref_freq of the non-existing "noisiest sample"
    //and we use the coverage of the weakest mutated sample to estimate the coverage of the "noisiest sample"
    else{
      my_mplp->cleanliness=min_other_ref_freq_limit; //use the highest possible noise level (strict)
      my_mplp->mut_score = -log10(fisher22((uint32_t) ((1-my_mplp->mut_freq) * my_mplp->cov_in_weakest),
                             (uint32_t) (my_mplp->mut_freq * my_mplp->cov_in_weakest),
                             (uint32_t) (min_other_ref_freq_limit * my_mplp->counts[weakest_idx][COV_IDX]), // use the coverage of the weakest mutated sample
                             (uint32_t) ((1-min_other_ref_freq_limit) * my_mplp->counts[weakest_idx][COV_IDX]),1));

    }
    copy_mplp(&saved_mutations[*mut_ptr],my_mplp);
    (*mut_ptr)++;
    return 1;
    // return 0;
}


/*
    gets the highest not reference mut freq
*/
int get_max_non_ref_freq(struct mplp* my_mplp,double* val, int* idx, char* mut_base){

    //arrays only for encoding/decoding base indices to bases
    int base_2_idx[256];
    base_2_idx[ 'A' ] = A_IDX;base_2_idx[ 'C' ] = C_IDX;
    base_2_idx[ 'G' ] = G_IDX;base_2_idx[ 'T' ] = T_IDX;
    char idx_2_base[MAX_IDX];
    idx_2_base[ A_IDX ] = 'A';idx_2_base[ C_IDX ] = 'C';
    idx_2_base[ G_IDX ] = 'G';idx_2_base[ T_IDX ] = 'T';

    //get index for ref base
    int ref_idx=base_2_idx[(int)my_mplp->ref_nuq];

    //initialize values to negative numbers
    *val = -42;
    *idx  = -42;

    int i,j;
    //loop over samples
    for(i=0;i<my_mplp->n_samples;i++){
        //loop over bases
        for(j=A_IDX;j<T_IDX+1;j++){
            if ( j!=ref_idx && //base is not the reference base
                my_mplp->freqs[i][j] > *val && //larger than largest yet
                my_mplp->freqs[i][j] != ZERO_COV_FREQ ){ //not a 0 cov sample
                //save value of max, sample idx, and the mut base as chr
                *val=my_mplp->freqs[i][j];
                *idx=i;
                *mut_base=idx_2_base[j];
            }
        }
    }
    return 0;
}


  /*
      - checks if ANY of the samples are noisy at the given position
      - checks for the weakest mutated and the noisiest clean sample:
          - noisiest_idx
          - weakest_idx
  */

  int check_if_noisy(struct mplp* my_mplp, int ref_idx, int alt_idx, double sample_mut_freq_limit,double min_other_ref_freq_limit, int cov_limit,
                    int* current_ploidy, int* noisiest_idx, int* weakest_idx, int shared_by_all){

      int i;
      double lowest_mutfreq = 10;
      double lowest_rnf = 10;
      *noisiest_idx = -42;
      *weakest_idx = -42;
      int number_of_cleans = 0;
      my_mplp->mutated_or_not = malloc(my_mplp->n_samples * sizeof(int));

      //loop over samples to find if there are noisy ones and IF NOT, then which samples are the weakest and the noisiest ones
      for(i=0;i<my_mplp->n_samples;i++)
      {
        //////////////////////////////////////////
        // printf("sample number: %d\n", i);
        //////////////////////////////////////////

        my_mplp->mutated_or_not[i] = 0;
        if (my_mplp->freqs[i][alt_idx] < sample_mut_freq_limit*2/current_ploidy[i] && // not really mutated
            my_mplp->freqs[i][ref_idx] < min_other_ref_freq_limit && // not really clean
            my_mplp->freqs[i][alt_idx] >=0 && my_mplp->freqs[i][ref_idx] >=0) // but definitely covered - esetleg itt lehet gond?
        {
          //////////////////////////////////////////
          // printf("found noisy, quitting!\n");
          //////////////////////////////////////////

          return 1; //found a noisy sample, ignore position
        }
        else if (my_mplp->freqs[i][ref_idx] > min_other_ref_freq_limit && // clean
                 my_mplp->freqs[i][alt_idx] < sample_mut_freq_limit*2/current_ploidy[i] && // not really mutated
                 my_mplp->freqs[i][alt_idx] >=0 && my_mplp->freqs[i][ref_idx] >=0){ //and covered
                   number_of_cleans++;
                   if (my_mplp->freqs[i][ref_idx] < lowest_rnf){ //check if noisier than the noisiest so far
                     *noisiest_idx=i;
                     lowest_rnf=my_mplp->freqs[i][ref_idx];
                   }
                   //////////////////////////////////////////
                   // printf("found clean!\n");
                   //////////////////////////////////////////
                 }
        else if (my_mplp->freqs[i][alt_idx] >= sample_mut_freq_limit*2/current_ploidy[i] && //sample mutated
                 my_mplp->counts[i][COV_IDX] >= cov_limit){ //and covered sufficiently
          my_mplp->mutated_or_not[i] = 1; //set its mutation status to "yes"
          if (my_mplp->freqs[i][alt_idx] < lowest_mutfreq){ //check if weaker than the weakest so far
            *weakest_idx=i;
            lowest_mutfreq=my_mplp->freqs[i][alt_idx];
          }
          //////////////////////////////////////////
          // printf("found mutated!\n");
          //////////////////////////////////////////
        }
      }

      //if no weakest sample was found > no mutated samples at all, skip position
      if (*weakest_idx == -42)
      {
        //////////////////////////////////////////
        // printf("weakest ID not found...\n");
        //////////////////////////////////////////
        return 1;
      }
      //if no noisiest sample was found and also, no clean ones > all of the samples are mutated, special treatment
      if (*noisiest_idx == -42 && number_of_cleans == 0)
      {
        *noisiest_idx = -999;
        //////////////////////////////////////////
        // printf("noisiest ID not found, there are no clean samples\n");
        //////////////////////////////////////////
        if (shared_by_all == 1){return 0;}
        else{return 1;}
        // return 0; //this is the original version, but it's probably best if we just skip these
        // return 1;
      }
      //no noisiest sample was found but there ARE clean samples (this should not happen) > skip position
      else if (*noisiest_idx == -42 && number_of_cleans != 0)
      {
        //////////////////////////////////////////
        // printf("noisiest ID not found, but there are no clean samples\n");
        //////////////////////////////////////////
        return 1;
      }
      return 0;
  }

/*
    gets the lowest reference freq, except for 1 sample
*/
int get_min_ref_freq(struct mplp* my_mplp,double* val, int idx_2skip, int* other_idx ){
    //initialize value to large number
    *val = 42;
    *other_idx=0;

    if (my_mplp->n_samples == 1){
      *other_idx=idx_2skip;
      *val = 1;
      return 0;
    }
    int i;
    //loop over samples
    for(i=0;i<my_mplp->n_samples;i++){
        if ( i != idx_2skip && // skip the mutated sample
           my_mplp->freqs[i][REF_IDX] < *val && //smaller than smallest yet
           my_mplp->freqs[i][REF_IDX] != ZERO_COV_FREQ ){ //not a 0 cov sample
                //save value of min
            *val=my_mplp->freqs[i][REF_IDX];
            *other_idx=i;
        }
    }

    //all other samples had zero coverage:
    if (*val==42){
        return 1;
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////
// Call indels
////////////////////////////////////////////////////////////////////////////


/*
    calls indels
*/
int call_indel_with_pl(struct mplp* saved_mutations, int* mut_ptr, struct mplp* my_mplp,
               double sample_mut_freq_limit,double min_other_ref_freq_limit,int cov_limit,
               char* last_gap_chrom, int last_gap_pos_start,int last_gap_pos_end,
               int prox_gap_min_dist_SNV,int prox_gap_min_dist_indel,
               int *ploidy_id, int *next_pos, int *current_ploidy, int *max_number_of_ranges, struct ploidy ** pl_r_array,
               int unique_only, int default_ploidy, int shared_by_all){

    //filter position if it is too close to last gap
    if ( last_gap_chrom != NULL && //no gap yet
         strcmp(last_gap_chrom,my_mplp->chrom) == 0  && //same chrom
         my_mplp->pos - last_gap_pos_end < prox_gap_min_dist_indel  && // proximal gap
         my_mplp->pos != last_gap_pos_start ){ //  dont filter for indel because of himself!
        return 0;
    }

    //skip position if the reference base is N
    if (my_mplp->ref_nuq == 'N'){
      return 0;
    }

    double sample_indel_freq;
    int sample_idx;
    char mut_indel[MAX_INDEL_LEN];
    char mut_type[4];

    //check if at least one of the samples show an elevated frequency of indels
    get_max_indel_freq(my_mplp,&sample_indel_freq,&sample_idx,mut_indel,mut_type);

    //if all the samples appear to be clean, skip position from the start
    if (sample_indel_freq < 0.02){
      return 0;
    }

    //if we are only looking for unique mutations, it's faster
    if ((unique_only == 1 && sample_idx >= 0) || (my_mplp->n_samples == 1)){
      //update the ploidy of the "most mutated" sample
      update_ploidy(my_mplp, &sample_idx, ploidy_id, next_pos, current_ploidy, max_number_of_ranges, pl_r_array, default_ploidy);
      int cp = current_ploidy[sample_idx];
      int other_idx;
      double min_other_noindel_freq;

      //get ref_freq (or noindel_freq) in the noisiest other sample
      //note: here it is assumed, that in the "most mutated sample", only one indel is present, and we simply choose the first one to appear in the pileup file
      get_min_other_noindel_freq(my_mplp,&min_other_noindel_freq,sample_idx,&other_idx);

      if (sample_indel_freq >= (sample_mut_freq_limit)*2/cp && // indel freq larger than limit
          min_other_noindel_freq > min_other_ref_freq_limit && // noisiest other sample is clean enough
          my_mplp->counts[sample_idx][COV_IDX] >= cov_limit ){ //coverage higher than limit

          //update mutation information in mplp structure
          int i;
          strncpy(my_mplp->mut_indel,mut_indel,MAX_INDEL_LEN);
          my_mplp->mutated_or_not = malloc(my_mplp->n_samples * sizeof(int));
          for(i=0;i<my_mplp->n_samples;i++){my_mplp->mutated_or_not[i]=0;}
          my_mplp->mutated_or_not[sample_idx]=1;
          my_mplp->ploidy = cp;
          my_mplp->mut_freq=sample_indel_freq;
          my_mplp->cleanliness=min_other_noindel_freq;
          if (my_mplp->n_samples == 1){
            my_mplp->cleanliness = min_other_ref_freq_limit;
          }
          my_mplp->cov_in_weakest=my_mplp->counts[sample_idx][COV_IDX];
          strncpy(my_mplp->mut_type,mut_type,4);
          my_mplp->mut_score = -log10(fisher22((uint32_t) ((1-sample_indel_freq) * my_mplp->counts[sample_idx][COV_IDX]),
                                 (uint32_t) (sample_indel_freq * my_mplp->counts[sample_idx][COV_IDX]),
                                 (uint32_t) (my_mplp->cleanliness * my_mplp->counts[other_idx][COV_IDX]),
                                 (uint32_t) ((1-my_mplp->cleanliness) *my_mplp->counts[other_idx][COV_IDX]),1));

          //proximal hindsight filtering for SNVs before
          proximal_gap_hindsight_filter(saved_mutations,mut_ptr,my_mplp->chrom,
                                        my_mplp->pos,prox_gap_min_dist_SNV,prox_gap_min_dist_indel);

          //save potential mutation
          copy_mplp(&saved_mutations[*mut_ptr],my_mplp);
          (*mut_ptr)++;
      }
      return 0;
    }

    //if we are looking for non-unique mutations as well, it's more complicated:

    //updating ploidy for ALL of the samples
    update_ploidy_all(my_mplp, ploidy_id, next_pos, current_ploidy, max_number_of_ranges, pl_r_array, default_ploidy);

    //if the largest indel freq is still lower than the current limit, skip the position
    //(it is important to skip as many positions as possible without the lengthy evaluation process below)
    if (sample_indel_freq < (sample_mut_freq_limit)*2/current_ploidy[sample_idx]){
      return 0;
    }

    //search for the indel that is supported with the largest number of reads thoughout all the samples
    char most_frequent_indel[MAX_INDEL_LEN] = "noindel";
    int indel_not_found = 1;
    indel_not_found = get_most_frequent_indel(my_mplp,most_frequent_indel,mut_type);

    //if it could not be found, skip position
    if (indel_not_found == 1){
      return 0;
    }

    //given the above found most frequent indel, we search for noisy positions (mut_freq < sample_mut_freq_limit && ref_freq < min_other_ref_freq_limit &&  cov >= 0)
    //mut_freq is defined as the frequency OF THE MOST FREQUENT INDEL in the given sample
    //ref_freq is defined as the frequency of the reference bases that did not contain any indel

    double noisiest_reffreq;
    int noisiest_cov;
    int weakest_idx;
    double weakest_mutfreq;
    int weakest_cov;
    int is_indel_noisy = check_if_indel_noisy(my_mplp, sample_mut_freq_limit, min_other_ref_freq_limit, cov_limit,
                                    current_ploidy, &noisiest_reffreq, &noisiest_cov, &weakest_mutfreq, &weakest_cov, &weakest_idx,
                                  most_frequent_indel, mut_type, shared_by_all);

    //if at least one noisy sample was found, skip position
    if (is_indel_noisy == 1){
      return 0;
    }

    //otherwise update mutation information in mplp structure
    strncpy(my_mplp->mut_indel,most_frequent_indel,MAX_INDEL_LEN);
    my_mplp->ploidy = current_ploidy[weakest_idx];
    my_mplp->mut_freq=weakest_mutfreq;
    my_mplp->cov_in_weakest=weakest_cov;
    strncpy(my_mplp->mut_type,mut_type,4);
    if (noisiest_cov != -999){
      my_mplp->cleanliness=noisiest_reffreq;
      my_mplp->mut_score = -log10(fisher22((uint32_t) ((1-my_mplp->mut_freq) * my_mplp->cov_in_weakest),
                             (uint32_t) (my_mplp->mut_freq * my_mplp->cov_in_weakest),
                             (uint32_t) (my_mplp->cleanliness * noisiest_cov),
                             (uint32_t) ((1-my_mplp->cleanliness) * noisiest_cov),1));

    }
    //if no clean samples were found (meaning that all of the covered samples are mutated)
    //we use the highest possible noise level (min_other_ref_freq_limit) to estimate the ref_freq of the non-existing "noisiest sample"
    //and we use the coverage of the weakest mutated sample to estimate the coverage of the "noisiest sample"
    else{
      my_mplp->cleanliness=min_other_ref_freq_limit; //use the highest possible noise level (this is strict)
      my_mplp->mut_score = -log10(fisher22((uint32_t) ((1-my_mplp->mut_freq) * my_mplp->cov_in_weakest),
                             (uint32_t) (my_mplp->mut_freq * my_mplp->cov_in_weakest),
                             (uint32_t) (min_other_ref_freq_limit * my_mplp->cov_in_weakest), // use the coverage of the weakest mutated sample
                             (uint32_t) ((1-min_other_ref_freq_limit) * my_mplp->cov_in_weakest),1));

    }

    //proximal hindsight filtering for SNVs before
    proximal_gap_hindsight_filter(saved_mutations,mut_ptr,my_mplp->chrom,
                                  my_mplp->pos,prox_gap_min_dist_SNV,prox_gap_min_dist_indel);

    copy_mplp(&saved_mutations[*mut_ptr],my_mplp);
    (*mut_ptr)++;

    return 0;
}


/*
    gets the highest indel freq
*/
int get_max_indel_freq(struct mplp* my_mplp,double* val, int* idx, char* mut_indel,char* mut_type){
    //initialize values to negative numbers
    *val = 0 ;
    *idx  = 0 ;

    int i,j;
    //loop over samples
    for(i=0;i<my_mplp->n_samples;i++){
        if( my_mplp->freqs[i][INS_START_IDX] > (*val) && //larger than largest yet
                    my_mplp->freqs[i][INS_START_IDX] != ZERO_COV_FREQ ){ //not a 0 cov sample
            //save value of max, sample idx, and the mut base as chr
            *val=my_mplp->freqs[i][INS_START_IDX];
            *idx=i;
            //copy first indel, and make it uppercase
            for(j=0;j<= (int) strlen(my_mplp->ins_bases[i][0]);j++){
                mut_indel[j] = (char) toupper(my_mplp->ins_bases[i][0][j]);
            }
            strncpy(mut_type,"INS\0",4);
        }
        if( my_mplp->freqs[i][DEL_START_IDX] > (*val) && //larger than largest yet
                    my_mplp->freqs[i][DEL_START_IDX] != ZERO_COV_FREQ ){ //not a 0 cov sample
            //save value of max, sample idx, and the mut base as chr
            *val=my_mplp->freqs[i][DEL_START_IDX];
            *idx=i;
            //copy first indel, and make it uppercase
            for(j=0;j<= (int) strlen(my_mplp->del_bases[i][0]);j++){
                mut_indel[j] = (char) toupper(my_mplp->del_bases[i][0][j]);
            }
            strncpy(mut_type,"DEL\0",4);
        }
    }
    return 0;
}

/*
    gets the indel that is supported by the largest number of reads in all of the samples combined
*/

int get_most_frequent_indel(struct mplp* my_mplp,char* most_frequent_indel,char* mut_type){
    int i,j;
    int num_of_muts=0;

    //loop over samples to find the maximal total number of different insertions or deletions
    for(i=0;i<my_mplp->n_samples;i++){
      if(strcmp(mut_type,"INS") ==0){
        num_of_muts = num_of_muts + my_mplp->counts[i][INS_START_IDX];
      }
      else if (strcmp(mut_type,"DEL") ==0){
        num_of_muts = num_of_muts + my_mplp->counts[i][DEL_START_IDX];
      }
    }

    // allocate arrays of length num_of_muts, one for storing indel strings and one for storing their counts
    char** all_indel_types = (char**) malloc( (num_of_muts) * sizeof(char*));
    for (i=0;i<num_of_muts;i++){all_indel_types[i] = (char*) malloc( (MAX_INDEL_LEN+1) * sizeof(char)); strcpy(all_indel_types[i], "nothing");}
    int * all_indel_types_count = (int *)malloc(num_of_muts * sizeof(int));
    for (i=0;i<num_of_muts;i++){all_indel_types_count[i] = 0;}

    int k, current_idx;
    current_idx=0;
    int found_it = 0;
    char * tmp_indel = (char*) malloc( (MAX_INDEL_LEN+1) * sizeof(char));
    strcpy(tmp_indel, "nothing");

    for(i=0;i<my_mplp->n_samples;i++){
      if(strcmp(mut_type,"INS") ==0 && my_mplp->counts[i][INS_START_IDX] > 0){
        for (j=0;j<my_mplp->counts[i][INS_START_IDX];j++){
          found_it = 0;

          //convert current indel to uppercase
          strcpy(tmp_indel, my_mplp->ins_bases[i][j]);
          for (k=0; k<= (int) strlen(tmp_indel); k++){
            tmp_indel[k] = (char) toupper(tmp_indel[k]);
          }

          //check if it's already in the counter list, if it is, increase its value by 1
          for (k=0;k<current_idx+1;k++){
            if (strcmp(all_indel_types[k], tmp_indel) == 0){
              all_indel_types_count[k]++;
              found_it = 1;
            }
          }

          // if it's not in the list already, add it to it and set its counter value to 1
          if (found_it == 0){
            strcpy(all_indel_types[current_idx], tmp_indel);
            all_indel_types_count[current_idx]++;
            current_idx++;
          }
        }
      }
      else if(strcmp(mut_type,"DEL") ==0 && my_mplp->counts[i][DEL_START_IDX] > 0){
        for (j=0;j<my_mplp->counts[i][DEL_START_IDX];j++){
          found_it = 0;

          //convert current indel to uppercase
          strcpy(tmp_indel, my_mplp->del_bases[i][j]);
          for (k=0; k<= (int) strlen(tmp_indel); k++){
            tmp_indel[k] = (char) toupper(tmp_indel[k]);
          }

          //check if it's already in the counter list, if it is, increase its value by 1
          for (k=0;k<current_idx;k++){
            if (strcmp(all_indel_types[k], tmp_indel) == 0){
              all_indel_types_count[k]++;
              found_it = 1;
            }
          }

          // if it's not in the list already, add it to it and set its counter value to 1
          if (found_it == 0){
            strcpy(all_indel_types[current_idx], tmp_indel);
            all_indel_types_count[current_idx]++;
            current_idx++;
          }
        }
      }
    }

    //find the indel with the largest count value
    int max_value = 0;
    for (i=0; i<num_of_muts;i++){
      if (all_indel_types_count[i] > max_value){
        max_value = all_indel_types_count[i];
        strcpy(most_frequent_indel, all_indel_types[i]);
      }
    }

    // free resources
    for (i=0;i<num_of_muts;i++){free(all_indel_types[i]);}
    free(all_indel_types);
    free(all_indel_types_count);
    free(tmp_indel);

    //if no indels found, return 1
    if (max_value == 0){
      return 1;
    }

    return 0;
}

/*
    check if there are any noisy samples for indel calling (use the previously determined value of the most frequent indel)
*/

int check_if_indel_noisy(struct mplp* my_mplp, double sample_mut_freq_limit, double min_other_ref_freq_limit, int cov_limit,
                                int* current_ploidy, double* noisiest_reffreq, int* noisiest_cov, double* weakest_mutfreq, int* weakest_cov, int* weakest_id,
                                char* most_frequent_indel, char* mut_type, int shared_by_all){
    //get index for ref base
    int base_2_idx[256];
    base_2_idx[ 'A' ] = A_IDX;base_2_idx[ 'C' ] = C_IDX;
    base_2_idx[ 'G' ] = G_IDX;base_2_idx[ 'T' ] = T_IDX;
    int ref_idx=base_2_idx[(int)my_mplp->ref_nuq];

    int i, j, k;
    *noisiest_reffreq = 10;
    *weakest_mutfreq = 10;
    *weakest_cov = -42;
    *noisiest_cov = -42;
    *weakest_id = -42;

    double current_mut_freq = -42;
    double current_ref_freq = -42;

    int number_of_cleans = 0;
    int number_of_muts = 0;
    char * tmp_indel = (char*) malloc( (MAX_INDEL_LEN+1) * sizeof(char));
    strcpy(tmp_indel, "nothing");

    my_mplp->mutated_or_not = malloc(my_mplp->n_samples * sizeof(int));

    //loop over samples to find if there are any noisy ones and IF NOT, then which samples are the weakest mutated and the noisiest clean ones
    for(i=0;i<my_mplp->n_samples;i++)
    {
      //determine mut_freq and ref_freq first
      if(strcmp(mut_type,"INS") ==0 && my_mplp->counts[i][INS_START_IDX] > 0){
        number_of_muts = 0;
        for (j=0; j<my_mplp->counts[i][INS_START_IDX]; j++){
          strcpy(tmp_indel, my_mplp->ins_bases[i][j]);
          for (k=0; k<= (int) strlen(tmp_indel); k++){
            tmp_indel[k] = (char) toupper(tmp_indel[k]);
          }
          if (strcmp(tmp_indel, most_frequent_indel) == 0){
            number_of_muts++;
          }
        }
        current_mut_freq = (double) number_of_muts/my_mplp->counts[i][COV_IDX];
      }
      else if(strcmp(mut_type,"DEL") ==0 && my_mplp->counts[i][DEL_START_IDX] > 0){
        number_of_muts = 0;
        for (j=0; j<my_mplp->counts[i][DEL_START_IDX]; j++){
          strcpy(tmp_indel, my_mplp->del_bases[i][j]);
          for (k=0; k<= (int) strlen(tmp_indel); k++){
            tmp_indel[k] = (char) toupper(tmp_indel[k]);
          }
          if (strcmp(tmp_indel, most_frequent_indel) == 0){
            number_of_muts++;
          }
        }
        current_mut_freq = (double) number_of_muts/my_mplp->counts[i][COV_IDX];
      }
      else{
        current_mut_freq = 0;
      }
      current_ref_freq = my_mplp->freqs[i][ref_idx] - my_mplp->freqs[i][INS_START_IDX] - my_mplp->freqs[i][DEL_START_IDX];

      my_mplp->mutated_or_not[i] = 0;
      if (current_mut_freq < sample_mut_freq_limit*2/current_ploidy[i] && // not really mutated
          current_ref_freq < min_other_ref_freq_limit && // not really clean
          my_mplp->counts[i][COV_IDX] > 0) // but definitely covered
      {
        return 1; //found a non-clean sample, skip position
      }
      else if (current_ref_freq >= min_other_ref_freq_limit && // clean
               my_mplp->counts[i][COV_IDX] > 0){ //and covered
                 number_of_cleans++;
                 if (current_ref_freq < *noisiest_reffreq){ //check if noisiest than the noisiest so far
                   *noisiest_reffreq=current_ref_freq;
                   *noisiest_cov = my_mplp->counts[i][COV_IDX];
                 }
               }
      else if (current_mut_freq >= sample_mut_freq_limit*2/current_ploidy[i] && //sample mutated
        my_mplp->counts[i][COV_IDX] >= cov_limit){ //and covered sufficiently
        my_mplp->mutated_or_not[i] = 1; //set mutational status to "yes"
        if (current_mut_freq < *weakest_mutfreq){ //check if weaker than the weakest so far
          *weakest_mutfreq = current_mut_freq;
          *weakest_cov = my_mplp->counts[i][COV_IDX];
          *weakest_id = i;
        }
      }
    }

    //if could not find the weakest sample, then no mutated ones exist, skip position
    if (*weakest_cov <= 0)
    {
      return 1;
    }
    //if could not find the noisiest one and there are no clean samples at all, then all samples are mutated > special treatment
    if (*noisiest_cov <= 0 && number_of_cleans == 0)
    {
      *noisiest_cov = -999;
      if (shared_by_all == 1){return 0;}
      else{return 1;}
      // return 1;
      // return 0; this is the original version
    }
    //if could not find the noisiest, but there ARE clean ones, skip the position
    else if (*noisiest_cov <= 0 && number_of_cleans != 0)
    {
      return 1;
    }
    return 0;
}

/*
    gets the highest noindel freq, except for 1 sample
*/
int get_min_other_noindel_freq(struct mplp* my_mplp,double* val, int idx_2skip, int* other_idx ){
    //initialize value to  large number
    *val =  42;
    *other_idx=0;

    if(my_mplp->n_samples == 1){
      *val=1;
      *other_idx = idx_2skip;
      return 0;
    }

    int i;
    //loop over samples
    for(i=0;i<my_mplp->n_samples;i++){
        double noindel_freq= my_mplp->freqs[i][REF_IDX] - my_mplp->freqs[i][INS_START_IDX] -
                               my_mplp->freqs[i][DEL_START_IDX];
        if ( i != idx_2skip && // skip the mutated sample
               noindel_freq < *val  && // smaller
               my_mplp->freqs[i][INS_START_IDX] != ZERO_COV_FREQ ){ //not a 0 cov sample
            //save value of min
            *val=noindel_freq;
            *other_idx=i;
        }
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////
// Get ploidy estimation for different genomic ranges for different samples
////////////////////////////////////////////////////////////////////////////

/*
    check if file exists
*/

int check_for_ploidy_file(char* ploidy_filename){
  FILE *file;
  if ((file = fopen(ploidy_filename, "r")))
  {
    int lines;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    lines=0;
    while ((read = getline(&line, &len, file)) != -1) {
        lines++;
    }
    fclose(file);
    return lines-1;
  }
  return 0;
}

/*
    initialize ploidy structure
*/

int init_ploidy(struct ploidy* my_ploidy){
    my_ploidy->chrom=NULL;
    my_ploidy->pos_from=-42;
    my_ploidy->pos_to=-42;
    my_ploidy->ploidy_est=-42;
    return 0;
}

/*
    free ploidy structure
*/

int free_ploidy(struct ploidy* my_ploidy){
    if( my_ploidy->chrom !=NULL) free(my_ploidy->chrom);
    //initalize the pointers to null after freeing
    init_ploidy(my_ploidy);
    return 0;
}

/*
    counting number of sample names in sample string (comma-separated)
*/

int get_token_num (char* sample_string)
{
    char buf_tmp[1000];
    strcpy(buf_tmp, sample_string);
    char *p_tmp = strtok (buf_tmp, ", ");
    int array_size = 0;
    while (p_tmp != NULL)
    {
        array_size++;
        p_tmp = strtok (NULL, ", ");
    }
    return array_size;
}

/*
    storing samples names in a predefined array
*/

int fill_array (char * pointer_to_array[], char* sample_string)
{
    int i = 0;
    char *p = strtok (sample_string, ", ");
    while (p != NULL)
    {
        pointer_to_array[i++] = p;
        p = strtok (NULL, ", ");
    }
    return 0;
}


/*
    getting basic ploidy info from summary file (number of different ploidy files used, maximal number of ranges in a single file)
*/

int get_basic_ploidy_info(char* ploidy_info_filename, int * number_of_ploidy_files, int * max_number_of_ranges){
    int lines;
    int i=0;
    int k;
    FILE *file_ploidy_info;
    FILE *file_ploidy_ranges_current;
    file_ploidy_info = fopen(ploidy_info_filename, "r");
    char * line_ploidy_info = NULL;
    size_t len_ploidy_info = 0;
    ssize_t read_ploidy_info;

    char * line_ploidy_ranges_current = NULL;
    size_t len_ploidy_ranges_current = 0;

    while ((read_ploidy_info = getline(&line_ploidy_info, &len_ploidy_info, file_ploidy_info)) != -1) {
        if (i>0)
        {
            (*number_of_ploidy_files)++;
            char* tmp_str=NULL;
            ssize_t j=0;
            while(j<read_ploidy_info){
                //filepath
                get_next_entry(line_ploidy_info,read_ploidy_info,&j,&tmp_str);
                k = 0;
                if ((file_ploidy_ranges_current = fopen(tmp_str, "r")))
                {
                    line_ploidy_ranges_current = NULL;
                    len_ploidy_ranges_current = 0;
                    ssize_t read_ploidy_ranges_current;
                    lines=0;
                    while ((read_ploidy_ranges_current = getline(&line_ploidy_ranges_current, &len_ploidy_ranges_current, file_ploidy_ranges_current)) != -1) {
                        if (k > 0)
                        {
                            lines++;
                        }
                        k++;
                    }
                    fclose(file_ploidy_ranges_current);
                }
                if (lines > *max_number_of_ranges)
                {
                    *max_number_of_ranges = lines;
                }
                get_next_entry(line_ploidy_info,read_ploidy_info,&j,&tmp_str);
            }
            free(tmp_str);
        }
        i++;
    }
    free(line_ploidy_info);
    free(line_ploidy_ranges_current);
    return 0;
}


/*
    storing ploidy information from all ploidy files in a 2D array, where each item is a ploidy struct
*/

int build_ploidy_ranges_array(char* ploidy_info_filename, struct ploidy ** pl_r_array, int * ploidy_id, int * next_pos_to_check, char** sample_names, int * total_sample_num){
    int i=0;
    int k=0;
    int m=0;
    int num_of_samples_in_category;
    FILE *file_ploidy_info;
    FILE *file_ploidy_ranges_current;
    file_ploidy_info = fopen(ploidy_info_filename, "r");
    char * line_ploidy_info = NULL;
    size_t len_ploidy_info = 0;
    ssize_t read_ploidy_info;

    char * line_ploidy_ranges_current = NULL;
    size_t len_ploidy_ranges_current = 0;

    while ((read_ploidy_info = getline(&line_ploidy_info, &len_ploidy_info, file_ploidy_info)) != -1) {
        if (i>0)
        {
            char* tmp_str=NULL;
            ssize_t j=0;
            while(j<read_ploidy_info){
                //filepath
                get_next_entry(line_ploidy_info,read_ploidy_info,&j,&tmp_str);
                if ((file_ploidy_ranges_current = fopen(tmp_str, "r")))
                {
                    line_ploidy_ranges_current = NULL;
                    len_ploidy_ranges_current = 0;
                    ssize_t read_ploidy_ranges_current;
                    k = 0;
                    while ((read_ploidy_ranges_current = getline(&line_ploidy_ranges_current, &len_ploidy_ranges_current, file_ploidy_ranges_current)) != -1) {
                        if (k>0)
                        {
                            char* tmp_str_current=NULL;
                            ssize_t l=0;
                            while(l<read_ploidy_ranges_current){
                                //chrom
                                get_next_entry(line_ploidy_ranges_current,read_ploidy_ranges_current,&l,&tmp_str_current);
                                pl_r_array[i-1][k-1].chrom = (char*) malloc((strlen(tmp_str_current)+1) * sizeof(char));
                                strcpy( pl_r_array[i-1][k-1].chrom, tmp_str_current);
                                //posfrom
                                get_next_entry(line_ploidy_ranges_current,read_ploidy_ranges_current,&l,&tmp_str_current);
                                pl_r_array[i-1][k-1].pos_from= (int) strtol(tmp_str_current,NULL,10);
                                //posto
                                get_next_entry(line_ploidy_ranges_current,read_ploidy_ranges_current,&l,&tmp_str_current);
                                pl_r_array[i-1][k-1].pos_to= (int) strtol(tmp_str_current,NULL,10);
                                //ploidy
                                get_next_entry(line_ploidy_ranges_current,read_ploidy_ranges_current,&l,&tmp_str_current);
                                pl_r_array[i-1][k-1].ploidy_est= (int) strtol(tmp_str_current,NULL,10);
                            }
                            free(tmp_str_current);
                        }
                        k++;
                    }
                    fclose(file_ploidy_ranges_current);
                }
                //sample data
                get_next_entry(line_ploidy_info,read_ploidy_info,&j,&tmp_str);
                num_of_samples_in_category = get_token_num(tmp_str);
                char *sample_array[num_of_samples_in_category];
                fill_array(sample_array, tmp_str);
                for (k=0;k<num_of_samples_in_category; k++)
                {
                    for (m=0;m<*total_sample_num;m++)
                    {
                        if (strcmp(sample_names[m], sample_array[k]) == 0)
                        {
                            ploidy_id[m] = i-1;
                            next_pos_to_check[m] = 1;
                        }
                    }
                }
                printf("\n");

            }
            free(tmp_str);
        }
        i++;
    }
    free(line_ploidy_info);
    free(line_ploidy_ranges_current);
    return 0;
}


/*
    updating ploidy information for a given sample at a given genomic position
*/

int update_ploidy(struct mplp* my_mplp, int *sample_id, int *ploidy_id, int *next_pos, int *current_ploidy, int *max_number_of_ranges,
                  struct ploidy ** pl_r_array, int default_ploidy){
    int i;
    int found_it=0;
    int found_min=INT_MAX;
    if (my_mplp->pos >= next_pos[(*sample_id)])
    {
        for (i=0;i<*max_number_of_ranges;i++)
        {
            //printf("i: %d\tchrom: %s\tpos_to: %d\tpos_from: %d\tploidy_est: %d\n", i, pl_r_array[ploidy_id[*sample_id]][i].chrom, pl_r_array[ploidy_id[*sample_id]][i].pos_to, pl_r_array[ploidy_id[*sample_id]][i].pos_from, pl_r_array[ploidy_id[*sample_id]][i].ploidy_est);
            if ((pl_r_array[ploidy_id[*sample_id]][i].chrom != NULL) && (strcmp(pl_r_array[ploidy_id[*sample_id]][i].chrom, my_mplp->chrom) == 0))
            {
              if ((my_mplp->pos >= pl_r_array[ploidy_id[*sample_id]][i].pos_from) && (my_mplp->pos <= pl_r_array[ploidy_id[*sample_id]][i].pos_to))
              {
                  current_ploidy[*sample_id] = pl_r_array[ploidy_id[*sample_id]][i].ploidy_est;
                  next_pos[*sample_id] = pl_r_array[ploidy_id[*sample_id]][i].pos_to+1;
                  found_it = 1;
              }
              if ((my_mplp->pos < pl_r_array[ploidy_id[*sample_id]][i].pos_from) && (found_min > pl_r_array[ploidy_id[*sample_id]][i].pos_from))
              {
                  found_min = pl_r_array[ploidy_id[*sample_id]][i].pos_from;
              }
           }
        }
        if (found_it == 0)
        {
            current_ploidy[*sample_id] = default_ploidy;
            next_pos[*sample_id] = found_min;
        }
    }
    return 0;
}

/*
    updating ploidy information for all samples
*/

int update_ploidy_all(struct mplp* my_mplp, int *ploidy_id, int *next_pos, int *current_ploidy, int *max_number_of_ranges,
                      struct ploidy ** pl_r_array, int default_ploidy){
    int i,j;
    int found_it;
    int found_min;
    for (j=0; j<my_mplp->n_samples;j++)
    {
      found_it = 0;
      found_min = INT_MAX;
      if (my_mplp->pos >= next_pos[j])
      {
          for (i=0;i<*max_number_of_ranges;i++)
          {
              //printf("i: %d\tchrom: %s\tpos_to: %d\tpos_from: %d\tploidy_est: %d\n", i, pl_r_array[ploidy_id[*sample_id]][i].chrom, pl_r_array[ploidy_id[*sample_id]][i].pos_to, pl_r_array[ploidy_id[*sample_id]][i].pos_from, pl_r_array[ploidy_id[*sample_id]][i].ploidy_est);
              if ((pl_r_array[ploidy_id[j]][i].chrom != NULL) && (strcmp(pl_r_array[ploidy_id[j]][i].chrom, my_mplp->chrom) == 0))
              {
                if ((my_mplp->pos >= pl_r_array[ploidy_id[j]][i].pos_from) && (my_mplp->pos <= pl_r_array[ploidy_id[j]][i].pos_to))
                {
                    current_ploidy[j] = pl_r_array[ploidy_id[j]][i].ploidy_est;
                    next_pos[j] = pl_r_array[ploidy_id[j]][i].pos_to+1;
                    found_it = 1;
                }
                if ((my_mplp->pos < pl_r_array[ploidy_id[j]][i].pos_from) && (found_min <= pl_r_array[ploidy_id[j]][i].pos_from))
                {
                    found_min = pl_r_array[ploidy_id[j]][i].pos_from;
                }
              }
          }
          if (found_it == 0)
          {
            current_ploidy[j] = default_ploidy;
            next_pos[j] = found_min;
          }
      }
    }
    return 0;
}


/*
    read ploidy data from file
*/

int load_ploidy_file_to_array(char* ploidy_filename, struct ploidy * ploidy_array){
    int i=0;
    FILE *file;
    file = fopen(ploidy_filename, "r");
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    while ((read = getline(&line, &len, file)) != -1) {
        if (i>0)
        {
            char* tmp_str=NULL;
            ssize_t j=0;
            while(j<read){
                //chrom
                get_next_entry(line,read,&j,&tmp_str);
                ploidy_array[i-1].chrom = (char*) malloc((strlen(tmp_str)+1) * sizeof(char));
                strcpy( ploidy_array[i-1].chrom, tmp_str);
                //position from
                get_next_entry(line,read,&j,&tmp_str);
                ploidy_array[i-1].pos_from= (int) strtol(tmp_str,NULL,10);
                //position to
                get_next_entry(line,read,&j,&tmp_str);
                ploidy_array[i-1].pos_to= (int) strtol(tmp_str,NULL,10);
                //ploidy
                get_next_entry(line,read,&j,&tmp_str);
                ploidy_array[i-1].ploidy_est= (int) strtol(tmp_str,NULL,10);
            }
            free(tmp_str);
        }
        i++;
    }
    fclose(file);
    free(line);
    return 0;
}

/*
    look up ploidy for specific position
*/

int find_ploidy_idx(struct ploidy * ploidy_array,int n, char *ch, int pos)
{
  int i;
  for(i=0;i<n;i++)
  {
    if ((strcmp(ploidy_array[i].chrom,ch) == 0) && (ploidy_array[i].pos_from <= pos) && (ploidy_array[i].pos_to >= pos))
    {
        return i;
    }
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////
// Raw estimation of haploid coverage
////////////////////////////////////////////////////////////////////////////

/*
    check if position should be printed and adjust diploid and triploid data if necesarry
*/

int process_position_for_ploidy_est(struct mplp* my_mplp,
             double min_noise,int min_cov,int max_cov,
             int* dip_cov_total, int* dip_count, int* trip_cov_total, int* trip_count){

    double noise_freq;
    int sample_idx;
    char mut_base = 'E';

    get_max_non_ref_freq(my_mplp,&noise_freq,&sample_idx,&mut_base);

    if (noise_freq >= min_noise &&
        noise_freq <= (1-min_noise) &&
        my_mplp->counts[sample_idx][COV_IDX] >= min_cov &&
        my_mplp->counts[sample_idx][COV_IDX] <= max_cov){
          printf("%s\t%d\t%d\t%f\n", my_mplp->chrom, my_mplp->pos, my_mplp->counts[sample_idx][COV_IDX], noise_freq);
          if (noise_freq > DIP_MIN_FREQ &&
              noise_freq < DIP_MAX_FREQ){
                (*dip_cov_total) = (*dip_cov_total) + my_mplp->counts[sample_idx][COV_IDX];
                (*dip_count)++;
          }
          if ((noise_freq > TRIP_MIN_FREQ_LOW &&
              noise_freq < TRIP_MAX_FREQ_LOW) ||
              (noise_freq > TRIP_MIN_FREQ_HIGH &&
              noise_freq < TRIP_MAX_FREQ_HIGH)){
                (*trip_cov_total) = (*trip_cov_total) + my_mplp->counts[sample_idx][COV_IDX];
                (*trip_count)++;
          }
    }

    return 0;
}

int build_window_data(double ** window_data, char ** ch, int * pointer_wd, struct mplp* my_mplp){

    double noise_freq;
    int sample_idx;
    char mut_base = 'E';

    get_max_non_ref_freq(my_mplp,&noise_freq,&sample_idx,&mut_base);

    strcpy(ch[(*pointer_wd)], my_mplp->chrom); //chrom to chrom array
    window_data[(*pointer_wd)][0] = (double) my_mplp->pos*1.0; //pos
    window_data[(*pointer_wd)][1] = (double) my_mplp->counts[sample_idx][COV_IDX]*1.0; //local cov
    window_data[(*pointer_wd)][2] = noise_freq; //local noise_freq
    if (my_mplp->ref_nuq == 'C' || my_mplp->ref_nuq == 'G'){
        window_data[(*pointer_wd)][3] = 1.0; //local CorGness
    }
    else{
        window_data[(*pointer_wd)][3] = 0.0; //local CorGness
    }

    window_data[(*pointer_wd)][4] = 0.0; //total cov //if there is any data, reinitialize with 0, otherwise it stays -42
    window_data[(*pointer_wd)][5] = 0.0; //total CorGness
    window_data[(*pointer_wd)][6] = 0.0; //number of estimates

    (*pointer_wd)++;

    return 0;
}

int shift_window(double ** window_data, char ** ch, int * pointer_wd, int ws, int rows, int shift,
             double min_noise, int print_every_nth){

    int i, j;
    i = 0;
    j = 0;

    double sum_cov, db, avg_cov;
    sum_cov = 0;
    db = 0;
    avg_cov = 0;

    double sum_CorG, avg_CorG;
    sum_CorG = 0;
    avg_CorG = 0;

    //calculate average cov and CorGness for window
    for (i=0;i<ws;i++)
    {
      if (window_data[i][0] > 0) //valid position
      {
        sum_cov = sum_cov + window_data[i][1];
        sum_CorG = sum_CorG + window_data[i][3];
        db = db + 1;
      }
    }
    avg_cov = sum_cov/db;
    avg_CorG = sum_CorG/db;
    // printf("avg_cov: %f\n", avg_cov);

    //write avg_cov and avg_CorG to window_data, where there is a valid position
    for(i=0;i<ws;i++)
    {
      if (window_data[i][0] > 0) //valid position
      {
        window_data[i][4] = window_data[i][4] + avg_cov;
        window_data[i][5] = window_data[i][5] + avg_CorG;
        window_data[i][6] = window_data[i][6] + 1;
      }
    }

    //shift window: print those positions from the shifted that meet the predefined criteria, shift the otherwise
    for (i=0;i<ws;i++)
    {
      // if (i<shift && window_data[i][2] > min_noise && window_data[i][2] < 1-min_noise) //these are filtered, then printed, diploid and triploid cov data updated
      if (i<shift && ((int) window_data[i][0] % print_every_nth == 0 || (window_data[i][2] > min_noise && window_data[i][2] < 1-min_noise))) //these are filtered, then printed, diploid and triploid cov data updated
      {
          printf("%s\t%d\t%f\t%f\t%f\n", ch[i], (int) window_data[i][0], window_data[i][4]/window_data[i][6], window_data[i][2], window_data[i][5]/window_data[i][6]);
      }
      if (i<ws-shift)
      {
        for(j=0;j<rows;j++)
        {
          window_data[i][j]=window_data[i+shift][j];
          strcpy(ch[i], ch[i+shift]);
        }
      }
      if (i>=ws-shift)
      {
        for(j=0;j<rows;j++)
        {
          window_data[i][j]=-42;
        }
      }
    }
    (*pointer_wd) = ws-shift;

    return 0;
}

int print_last_window(double ** window_data, char ** ch, int ws,
             double min_noise, int print_every_nth){

    int i;
    i = 0;

    double sum_cov, db, avg_cov;
    sum_cov = 0;
    db = 0;
    avg_cov = 0;

    double sum_CorG, avg_CorG;
    sum_CorG = 0;
    avg_CorG = 0;

    //calculate average cov for window
    for (i=0;i<ws;i++)
    {
      if (window_data[i][0] > 0) //valid position
      {
        sum_cov = sum_cov + window_data[i][1];
        sum_CorG = sum_CorG + window_data[i][3];
        db = db + 1;
      }
    }
    avg_cov = sum_cov/db;
    avg_CorG = sum_CorG/db;

    //write avg_cov to window_data, where there is a valid position
    for(i=0;i<ws;i++)
    {
      if (window_data[i][0] > 0) //valid position
      {
        window_data[i][4] = window_data[i][4] + avg_cov;
        window_data[i][5] = window_data[i][5] + avg_CorG;
        window_data[i][6] = window_data[i][6] + 1;
      }
    }

    //print all positions meeting the criteria
    for (i=0;i<ws;i++)
    {
      // if (window_data[i][0] > 0 && window_data[i][2] > min_noise && window_data[i][2] < 1-min_noise)
      if (window_data[i][0] > 0 && ((int) window_data[i][0] % print_every_nth == 0 || (window_data[i][2] > min_noise && window_data[i][2] < 1-min_noise))) //these are filtered, then printed, diploid and triploid cov data updated
      {
        printf("%s\t%d\t%f\t%f\t%f\n", ch[i], (int) window_data[i][0], window_data[i][4]/window_data[i][6], window_data[i][2], window_data[i][5]/window_data[i][6]);
      }
    }
    return 0;
}
