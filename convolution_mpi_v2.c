//
//  convolution.c
//
//
//  Created by Josep Lluis Lerida on 11/03/15.
//
// This program calculates the convolution for PPM images.
// The program accepts an PPM image file, a text definition of the kernel matrix and the PPM file for storing the convolution results.
// The program allows to define image partitions for processing large images (>500MB)
// The 2D image is represented by 1D vector for chanel R, G and B. The convolution is applied to each chanel separately.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

// Estructura per emmagatzemar el contingut d'una imatge.
struct imagenppm{
    int altura;
    int ancho;
    char *comentario;
    int maxcolor;
    int P;
    int *R;
    int *G;
    int *B;
};
typedef struct imagenppm* ImagenData;

// Estructura per emmagatzemar el contingut d'un kernel.
struct structkernel{
    int kernelX;
    int kernelY;
    float *vkern;
};
typedef struct structkernel* kernelData;

static double mpi_time_accum = 0.0;
static double omp_time_accum = 0.0;

//Functions Definition
ImagenData initimage(char* nombre, FILE **fp, int partitions, int halo);
ImagenData duplicateImageData(ImagenData src, int partitions, int halo);

int readImage(ImagenData Img, FILE **fp, int dim, int halosize, long int *position);
int readImageFast(int *R, int *G, int *B, FILE *fp, long *position,
                  int pixel_count, int image_width, int halosize);
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position);
int savingChunk(ImagenData img, FILE **fp, int dim, int offset);
int convolve2D(int* inbuf, int* outbuf, int sizeX, int sizeY, float* kernel, int ksizeX, int ksizeY);
void freeImagestructure(ImagenData *src);
typedef struct {
    int partition_index;
    int chunk_height;
    int pixel_count;
    int output_offset;
    int valid_pixel_count;
} ChunkInfo;
void send_rgb_channels(const int *channel_r, const int *channel_g, const int *channel_b, int pixel_count, int peer_rank, int tag);
void recv_rgb_channels(int *channel_r, int *channel_g, int *channel_b, int pixel_count, int peer_rank, int tag, MPI_Status *status);
void send_chunk_info(const ChunkInfo *chunk_info, int peer_rank, int tag);
void recv_chunk_info(ChunkInfo *chunk_info, int peer_rank, int tag, MPI_Status *status);

//Open Image file and image struct initialization
ImagenData initimage(char* nombre, FILE **fp,int partitions, int halo){
    char c;
    char comentario[300];
    int i=0,chunk=0;
    ImagenData img=NULL;
    
    /*Se habre el fichero ppm*/

    if ((*fp=fopen(nombre,"r"))==NULL){
        perror("Error: ");
    }
    else{
        //Memory allocation
        img=(ImagenData) malloc(sizeof(struct imagenppm));

        //Reading the first line: Magical Number "P3"
        fscanf(*fp,"%c%d ",&c,&(img->P));
        
        //Reading the image comment
        while((c=fgetc(*fp))!= '\n'){comentario[i]=c;i++;}
        comentario[i]='\0';
        //Allocating information for the image comment
        img->comentario = calloc(strlen(comentario),sizeof(char));
        strcpy(img->comentario,comentario);
        //Reading image dimensions and color resolution
        fscanf(*fp,"%d %d %d",&img->ancho,&img->altura,&img->maxcolor);
        chunk = img->ancho*img->altura / partitions;
        //We need to read an extra row.
        chunk = chunk + img->ancho * halo;
        if ((img->R=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->G=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->B=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    }
    return img;
}

//Read the corresponding chunk from the source Image
int readImageFast(int *R, int *G, int *B, FILE *fp, long *position,
                  int pixel_count, int image_width, int halosize){
    if (fseek(fp, *position, SEEK_SET)) {
        perror("Error: ");
        return -1;
    }

    int haloposition = pixel_count - (image_width * halosize * 2);
    long start_position = *position;
    long buf_size = (long)pixel_count * 16 + 1024;
    char *buf = (char *)malloc(buf_size + 1);
    if (!buf) {
        return -1;
    }

    long bytes_read = fread(buf, 1, buf_size, fp);
    buf[bytes_read] = '\0';

    char *ptr = buf;
    for (int i = 0; i < pixel_count; i++) {
        if (halosize > 0 && i == haloposition) {
            *position = start_position + (long)(ptr - buf);
        }
        R[i] = (int)strtol(ptr, &ptr, 10);
        G[i] = (int)strtol(ptr, &ptr, 10);
        B[i] = (int)strtol(ptr, &ptr, 10);
    }

    if (halosize == 0) {
        *position = start_position + (long)(ptr - buf);
    }

    free(buf);
    return 0;
}

int readImage(ImagenData img, FILE **fp, int dim, int halosize, long *position){
    return readImageFast(img->R, img->G, img->B, *fp, position, dim, img->ancho, halosize);
}

// Open kernel file and reading kernel matrix. The kernel matrix 2D is stored in 1D format.
kernelData leerKernel(char* nombre){
    FILE *fp;
    int i=0;
    kernelData kern=NULL;
    
    /*Opening the kernel file*/
    fp=fopen(nombre,"r");
    if(!fp){
        perror("Error: ");
    }
    else{
        //Memory allocation
        kern=(kernelData) malloc(sizeof(struct structkernel));
        
        //Reading kernel matrix dimensions
        fscanf(fp,"%d,%d,", &kern->kernelX, &kern->kernelY);
        kern->vkern = (float *)malloc(kern->kernelX*kern->kernelY*sizeof(float));
        
        // Reading kernel matrix values
        for (i=0;i<(kern->kernelX*kern->kernelY)-1;i++){
            fscanf(fp,"%f,",&kern->vkern[i]);
        }
        fscanf(fp,"%f",&kern->vkern[i]);
        fclose(fp);
    }
    return kern;
}

// Open the image file with the convolution results
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position){
    /*Se crea el fichero con la imagen resultante*/
    if ( (*fp=fopen(nombre,"w")) == NULL ){
        perror("Error: ");
        return -1;
    }
    /*Writing Image Header*/
    fprintf(*fp,"P%d\n%s\n%d %d\n%d\n",img->P,img->comentario,img->ancho,img->altura,img->maxcolor);
    *position = ftell(*fp);
    return 0;
}

// Writing the image partition to the resulting file. dim is the exact size to write. offset is the displacement for avoid halos.
int savingChunk(ImagenData img, FILE **fp, int dim, int offset){
    int i,k=0;
    //Writing image partition
    for(i=offset;i<dim+offset;i++){
        fprintf(*fp,"%d %d %d ",img->R[i],img->G[i],img->B[i]);
//        if ((i+1)%6==0) fprintf(*fp,"\n");
        k++;
    }
//    printf ("Writed = %d pixels, dim=%d, offset=%d\n",k,dim, offset);
    return 0;
}

// This function free the space allocated for the image structure.
void freeImagestructure(ImagenData *src){
    
    free((*src)->comentario);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);
    
    free(*src);
}

int convolve2D(int* in, int* out, int dataSizeX, int dataSizeY,
               float* kernel, int kernelSizeX, int kernelSizeY)
{
    if (!in || !out || !kernel) return -1;
    if (dataSizeX <= 0 || kernelSizeX <= 0) return -1;

    int kCenterX = kernelSizeX / 2;
    int kCenterY = kernelSizeY / 2;

    int paddedW = dataSizeX + 2 * kCenterX;
    int paddedH = dataSizeY + 2 * kCenterY;

    int *padded = (int *) calloc(paddedW * paddedH, sizeof(int));
    if (!padded) return -1;

    for (int i = 0; i < dataSizeY; i++)
        for (int j = 0; j < dataSizeX; j++)
            padded[(i + kCenterY) * paddedW + (j + kCenterX)] = in[i * dataSizeX + j];

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dataSizeY; i++)
    for (int j = 0; j < dataSizeX; j++)
    {
        float sum = 0.0f;

        for (int m = 0; m < kernelSizeY; m++)
        for (int n = 0; n < kernelSizeX; n++)
        {
            int pr = i + kCenterY + (kCenterY - m);
            int pc = j + kCenterX + (kCenterX - n);
            sum += padded[pr * paddedW + pc] * kernel[m * kernelSizeX + n];
        }

        out[i * dataSizeX + j] = (int)(sum >= 0 ? sum + 0.5f : sum - 0.5f);
    }

    free(padded);
    return 0;
}

void print_usage(const char* program_name){
    printf("Usage: %s <image-file> <kernel-file> <result-file> <partitions>\n", program_name);
    printf("\n\nError, Missing parameters:\n");
    printf("format: ./serialconvolution image_file kernel_file result_file\n");
    printf("- image_file : source image path (*.ppm)\n");
    printf("- kernel_file: kernel path (text file with 1D kernel matrix)\n");
    printf("- result_file: result image path (*.ppm)\n");
    printf("- partitions : Image partitions\n\n");
}

#define WORK_TAG 1
#define RESULT_TAG 2
#define TERMINATE_TAG 3

typedef struct {
    int kernelX, kernelY;
    int ancho, altura, maxcolor;
} BcastHeader;


void send_rgb_channels(const int *channel_r, const int *channel_g, const int *channel_b, int pixel_count, int peer_rank, int tag) {
    double mpi_start = MPI_Wtime();
    MPI_Send(channel_r, pixel_count, MPI_INT, peer_rank, tag, MPI_COMM_WORLD);
    MPI_Send(channel_g, pixel_count, MPI_INT, peer_rank, tag, MPI_COMM_WORLD);
    MPI_Send(channel_b, pixel_count, MPI_INT, peer_rank, tag, MPI_COMM_WORLD);
    mpi_time_accum += MPI_Wtime() - mpi_start;
}

void recv_rgb_channels(int *channel_r, int *channel_g, int *channel_b, int pixel_count, int peer_rank, int tag, MPI_Status *status) {
    double mpi_start = MPI_Wtime();
    MPI_Recv(channel_r, pixel_count, MPI_INT, peer_rank, tag, MPI_COMM_WORLD, status);
    MPI_Recv(channel_g, pixel_count, MPI_INT, peer_rank, tag, MPI_COMM_WORLD, status);
    MPI_Recv(channel_b, pixel_count, MPI_INT, peer_rank, tag, MPI_COMM_WORLD, status);
    mpi_time_accum += MPI_Wtime() - mpi_start;
}

void send_rgb_packed(const int *r, const int *g, const int *b,
                     int pixel_count, int peer_rank, int tag) {
    int *buf = malloc(sizeof(int) * pixel_count * 3);
    memcpy(buf,                    r, sizeof(int) * pixel_count);
    memcpy(buf + pixel_count,      g, sizeof(int) * pixel_count);
    memcpy(buf + pixel_count * 2,  b, sizeof(int) * pixel_count);
    double mpi_start = MPI_Wtime();
    MPI_Send(buf, pixel_count * 3, MPI_INT, peer_rank, tag, MPI_COMM_WORLD);
    mpi_time_accum += MPI_Wtime() - mpi_start;
    free(buf);
}

void recv_rgb_packed(int *r, int *g, int *b,
                     int pixel_count, int peer_rank, int tag,
                     MPI_Status *status) {
    int *buf = malloc(sizeof(int) * pixel_count * 3);
    double mpi_start = MPI_Wtime();
    MPI_Recv(buf, pixel_count * 3, MPI_INT, peer_rank, tag, MPI_COMM_WORLD, status);
    mpi_time_accum += MPI_Wtime() - mpi_start;
    memcpy(r, buf,                    sizeof(int) * pixel_count);
    memcpy(g, buf + pixel_count,      sizeof(int) * pixel_count);
    memcpy(b, buf + pixel_count * 2,  sizeof(int) * pixel_count);
    free(buf);
}

void send_chunk_info(const ChunkInfo *chunk_info, int peer_rank, int tag) {
    double mpi_start = MPI_Wtime();
    MPI_Send((void *)chunk_info, 5, MPI_INT, peer_rank, tag, MPI_COMM_WORLD);
    mpi_time_accum += MPI_Wtime() - mpi_start;
}

void recv_chunk_info(ChunkInfo *chunk_info, int peer_rank, int tag, MPI_Status *status) {
    double mpi_start = MPI_Wtime();
    MPI_Recv(chunk_info, 5, MPI_INT, peer_rank, tag, MPI_COMM_WORLD, status);
    mpi_time_accum += MPI_Wtime() - mpi_start;
}

void precompute_chunk_info(ChunkInfo **chunks, int *valid_count,
                           long file_position, int total_chunks, 
                           int chunks_per_partition,
                           int image_height, int image_width,
                           int partition_count, int kernel_radius) {
    int partition_index;
    int chunk_index_in_partition;
    int partition_start_row;
    int partition_end_row;
    int partition_rows;
    int chunk_rows;
    int chunk_start_row;
    int valid_rows;
    int halo_top;
    int halo_bottom;
    int has_valid_chunk = 0;
    int chunk_count = 0;
    *chunks = (ChunkInfo *)malloc(sizeof(ChunkInfo) * total_chunks);
    for (int i = 0; i < total_chunks; i++) {
        partition_index = i / chunks_per_partition;
        chunk_index_in_partition = i % chunks_per_partition;
        partition_start_row = (partition_index * image_height) / partition_count;
        partition_end_row = ((partition_index + 1) * image_height) / partition_count;
        partition_rows = partition_end_row - partition_start_row;
        chunk_rows = (partition_rows + chunks_per_partition - 1) / chunks_per_partition;

        if (chunk_rows < 1) chunk_rows = 1;

        chunk_start_row = partition_start_row + (chunk_index_in_partition * chunk_rows);

        if (chunk_start_row >= partition_end_row) {
            continue;
        }

        valid_rows = partition_end_row - chunk_start_row;
        if (valid_rows > chunk_rows) valid_rows = chunk_rows;
        halo_top = (chunk_start_row == 0) ? 0 : kernel_radius;
        halo_bottom = (chunk_start_row + valid_rows >= image_height) ? 0 : kernel_radius;

        (*chunks)[chunk_count].partition_index = chunk_count;
        (*chunks)[chunk_count].chunk_height = valid_rows + halo_top + halo_bottom;
        (*chunks)[chunk_count].pixel_count = (*chunks)[chunk_count].chunk_height * image_width;
        (*chunks)[chunk_count].output_offset = halo_top * image_width;
        (*chunks)[chunk_count].valid_pixel_count = valid_rows * image_width;

        chunk_count++;
    }

    *valid_count = chunk_count;
}



int master(int argc, char **argv) {
    if(argc != 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int partition_count = atoi(argv[4]);
    int chunk_size, halo;
    long file_position = 0;
    double start, tstart=0, tend=0, tread=0, tcopy=0, tconv=0, tstore=0, treadk=0, tmpi=0;
    FILE *source_file=NULL,*result_file=NULL;
    ImagenData source_image=NULL, output_image=NULL;

    start = MPI_Wtime();
    tstart = start;
    kernelData kernel_data=NULL;
    if ( (kernel_data = leerKernel(argv[2]))==NULL) {
        return -1;
    }
    halo = (kernel_data->kernelY/2)*2;
    treadk = treadk + (MPI_Wtime() - start);

    start = MPI_Wtime();
    if ( (source_image = initimage(argv[1], &source_file, partition_count, halo)) == NULL) {
        return -1;
    }
    tread = tread + (MPI_Wtime() - start);

    long source_position = ftell(source_file);

    start = MPI_Wtime();
    if (initfilestore(source_image, &result_file, argv[3], &file_position)!=0) {
        perror("Error: ");
        return -1;
    }
    tstore = tstore + (MPI_Wtime() - start);

    MPI_Status status;
    int worker_rank;
    int mpi_world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    int worker_count = mpi_world_size - 1;
    int next_chunk_to_assign = 0;
    int chunks_per_partition = (worker_count * 2 > 4) ? worker_count * 2 : 4;
    int total_chunks = partition_count * chunks_per_partition;
    int kernel_radius = halo / 2;
    int terminated_workers = 0;
    if (worker_count <= 0) {
        return -1;
    }

    start = MPI_Wtime();
    BcastHeader hdr = {kernel_data->kernelX, kernel_data->kernelY,
                   source_image->ancho, source_image->altura, source_image->maxcolor};
    MPI_Bcast(&hdr, sizeof(BcastHeader), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(kernel_data->vkern, hdr.kernelX * hdr.kernelY, MPI_FLOAT, 0, MPI_COMM_WORLD);
    int image_path_len = (int)strlen(argv[1]) + 1;
    MPI_Bcast(&image_path_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(argv[1], image_path_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    mpi_time_accum += MPI_Wtime() - start;
    
    ChunkInfo *all_chunks;
    int valid_chunk_count;

    precompute_chunk_info(&all_chunks, &valid_chunk_count,
                      file_position, total_chunks, chunks_per_partition,
                      source_image->altura, source_image->ancho,
                      partition_count, kernel_radius);

    int max_chunk_pixels = 0;
    for (int c = 0; c < valid_chunk_count; c++) {
        if (all_chunks[c].pixel_count > max_chunk_pixels) {
            max_chunk_pixels = all_chunks[c].pixel_count;
        }
    }

    int *scan_r = NULL;
    int *scan_g = NULL;
    int *scan_b = NULL;
    long *chunk_byte_offsets = NULL;
    if (max_chunk_pixels > 0) {
        scan_r = (int *)malloc(sizeof(int) * max_chunk_pixels);
        scan_g = (int *)malloc(sizeof(int) * max_chunk_pixels);
        scan_b = (int *)malloc(sizeof(int) * max_chunk_pixels);
        chunk_byte_offsets = (long *)malloc(sizeof(long) * valid_chunk_count);
        if (!scan_r || !scan_g || !scan_b || !chunk_byte_offsets) {
            free(scan_r); free(scan_g); free(scan_b); free(chunk_byte_offsets);
            return -1;
        }
    }

    start = MPI_Wtime();
    long scan_position = source_position;
    for (int c = 0; c < valid_chunk_count; c++) {
        chunk_byte_offsets[c] = scan_position;
        if (readImageFast(scan_r, scan_g, scan_b, source_file, &scan_position,
                          all_chunks[c].pixel_count, source_image->ancho, kernel_radius)) {
            free(scan_r); free(scan_g); free(scan_b); free(chunk_byte_offsets);
            return -1;
        }
    }
    tread = tread + (MPI_Wtime() - start);
    free(scan_r); free(scan_g); free(scan_b);
    
    int *result_ready = calloc(valid_chunk_count, sizeof(int));
    int **resR = calloc(valid_chunk_count, sizeof(int*));
    int **resG = calloc(valid_chunk_count, sizeof(int*));
    int **resB = calloc(valid_chunk_count, sizeof(int*));
    ChunkInfo *result_infos = malloc(valid_chunk_count * sizeof(ChunkInfo));
    int next_to_write = 0;

    while (terminated_workers < worker_count) {
        start = MPI_Wtime();
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        mpi_time_accum += MPI_Wtime() - start;
        worker_rank = status.MPI_SOURCE;
        
        if(status.MPI_TAG == WORK_TAG) {
            start = MPI_Wtime();
            MPI_Recv(NULL, 0, MPI_BYTE, worker_rank, WORK_TAG, MPI_COMM_WORLD, &status);
            mpi_time_accum += MPI_Wtime() - start;
            if (next_chunk_to_assign < valid_chunk_count) {
                ChunkInfo *info = &all_chunks[next_chunk_to_assign];

                start = MPI_Wtime();
                send_chunk_info(info, worker_rank, WORK_TAG);
                MPI_Send(&chunk_byte_offsets[next_chunk_to_assign], 1, MPI_LONG,
                         worker_rank, WORK_TAG, MPI_COMM_WORLD);
                mpi_time_accum += MPI_Wtime() - start;
                next_chunk_to_assign++;
            } else {
                start = MPI_Wtime();
                MPI_Send(NULL, 0, MPI_BYTE, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
                mpi_time_accum += MPI_Wtime() - start;
                terminated_workers++;
            }
        } else if(status.MPI_TAG == RESULT_TAG) {
            ChunkInfo chunk_info;
            recv_chunk_info(&chunk_info, worker_rank, RESULT_TAG, &status);
            int ci = chunk_info.partition_index;

            resR[ci] = malloc(sizeof(int) * chunk_info.pixel_count);
            resG[ci] = malloc(sizeof(int) * chunk_info.pixel_count);
            resB[ci] = malloc(sizeof(int) * chunk_info.pixel_count);
            recv_rgb_packed(resR[ci], resG[ci], resB[ci],
                            chunk_info.pixel_count, worker_rank, RESULT_TAG, &status);
            result_infos[ci] = chunk_info;
            result_ready[ci] = 1;

            while (next_to_write < valid_chunk_count && result_ready[next_to_write]) {
                ChunkInfo *wi = &result_infos[next_to_write];
                int off = wi->output_offset;
                for (int px = 0; px < wi->valid_pixel_count; px++) {
                    fprintf(result_file, "%d %d %d ",
                            resR[next_to_write][off + px],
                            resG[next_to_write][off + px],
                            resB[next_to_write][off + px]);
                }
                free(resR[next_to_write]);
                free(resG[next_to_write]);
                free(resB[next_to_write]);
                next_to_write++;
            }
        }
    }

    free(result_ready);
    free(resR); free(resG); free(resB);
    free(result_infos);
    free(chunk_byte_offsets);

    double worker_mpi_local = (mpi_rank == 0) ? 0.0 : mpi_time_accum;
    double worker_mpi_time = 0.0;

    MPI_Reduce(&omp_time_accum, &tconv, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&worker_mpi_local, &worker_mpi_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    fclose(source_file);
    fclose(result_file);

    tend = MPI_Wtime();

    printf("Imatge: %s\n", argv[1]);
    printf("ISizeX : %d\n", source_image->ancho);
    printf("ISizeY : %d\n", source_image->altura);
    printf("kSizeX : %d\n", kernel_data->kernelX);
    printf("kSizeY : %d\n", kernel_data->kernelY);
    printf("%.6lf seconds elapsed for Reading image file.\n", tread);
    printf("%.6lf seconds elapsed for copying image structure.\n", tcopy);
    printf("%.6lf seconds elapsed for Reading kernel matrix.\n", treadk);
    printf("%.6lf seconds elapsed for OpenMP convolution.\n", tconv);
    printf("%.6lf seconds elapsed for MPI communication on the master.\n", mpi_time_accum);
    printf("%.6lf seconds elapsed for MPI communication on workers (max).\n", worker_mpi_time);
    printf("%.6lf seconds elapsed for writing the resulting image.\n", tstore);
    printf("%.6lf seconds elapsed\n", tend-tstart);

    freeImagestructure(&source_image);

    return 0;
}

int worker(int argc, char **argv) {
    (void)argc;
    (void)argv;
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Status status;
    double start;

        start = MPI_Wtime();
        BcastHeader hdr;
        MPI_Bcast(&hdr, sizeof(BcastHeader), MPI_BYTE, 0, MPI_COMM_WORLD);
        float *vkern = malloc(hdr.kernelX * hdr.kernelY * sizeof(float));
        MPI_Bcast(vkern, hdr.kernelX * hdr.kernelY, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int image_path_len = 0;
        MPI_Bcast(&image_path_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        char *image_path = (char *)malloc((size_t)image_path_len);
        if (!image_path) {
            free(vkern);
            return -1;
        }
        MPI_Bcast(image_path, image_path_len, MPI_CHAR, 0, MPI_COMM_WORLD);
        mpi_time_accum += MPI_Wtime() - start;

        FILE *worker_fp = fopen(image_path, "r");
        free(image_path);
        if (!worker_fp) {
            perror("Error: ");
            free(vkern);
            return -1;
        }

    while(1){
        start = MPI_Wtime();
        MPI_Send(NULL, 0, MPI_BYTE, 0, WORK_TAG, MPI_COMM_WORLD);
        mpi_time_accum += MPI_Wtime() - start;
        start = MPI_Wtime();
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        mpi_time_accum += MPI_Wtime() - start;

        if(status.MPI_TAG == TERMINATE_TAG) {
            start = MPI_Wtime();
            MPI_Recv(NULL, 0, MPI_BYTE, 0, TERMINATE_TAG, MPI_COMM_WORLD, &status);
            mpi_time_accum += MPI_Wtime() - start;
            break;
        } else if(status.MPI_TAG == WORK_TAG) {
            ChunkInfo chunk_info;
            recv_chunk_info(&chunk_info, 0, WORK_TAG, &status);

            int pixel_count = chunk_info.pixel_count;
            int *source_r = (int *)malloc(sizeof(int) * pixel_count);
            int *source_g = (int *)malloc(sizeof(int) * pixel_count);
            int *source_b = (int *)malloc(sizeof(int) * pixel_count);
            int *result_r = (int *)malloc(sizeof(int) * pixel_count);
            int *result_g = (int *)malloc(sizeof(int) * pixel_count);
            int *result_b = (int *)malloc(sizeof(int) * pixel_count);
            if (source_r == NULL || source_g == NULL || source_b == NULL ||
                result_r == NULL || result_g == NULL || result_b == NULL) {
                free(source_r); free(source_g); free(source_b);
                free(result_r); free(result_g); free(result_b);
                free(vkern);
                return -1;
            }

            long chunk_offset = 0;
            start = MPI_Wtime();
            MPI_Recv(&chunk_offset, 1, MPI_LONG, 0, WORK_TAG, MPI_COMM_WORLD, &status);
            mpi_time_accum += MPI_Wtime() - start;

            if (readImageFast(source_r, source_g, source_b, worker_fp, &chunk_offset,
                              pixel_count, hdr.ancho, hdr.kernelY / 2)) {
                free(source_r); free(source_g); free(source_b);
                free(result_r); free(result_g); free(result_b);
                fclose(worker_fp);
                free(vkern);
                return -1;
            }

            start = MPI_Wtime();
            convolve2D(source_r, result_r, hdr.ancho, chunk_info.chunk_height, vkern, hdr.kernelX, hdr.kernelY);
            convolve2D(source_g, result_g, hdr.ancho, chunk_info.chunk_height, vkern, hdr.kernelX, hdr.kernelY);
            convolve2D(source_b, result_b, hdr.ancho, chunk_info.chunk_height, vkern, hdr.kernelX, hdr.kernelY);
            omp_time_accum += MPI_Wtime() - start;

            send_chunk_info(&chunk_info, 0, RESULT_TAG);
            send_rgb_packed(result_r, result_g, result_b, pixel_count, 0, RESULT_TAG);

            free(source_r); free(source_g); free(source_b);
            free(result_r); free(result_g); free(result_b);
        }
    }

    double dummy = 0.0;
    MPI_Reduce(&omp_time_accum, &dummy, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double worker_mpi_local = mpi_time_accum;
    MPI_Reduce(&worker_mpi_local, &dummy, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    fclose(worker_fp);
    free(vkern);
    
    return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int mpi_rank, mpi_world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    (void)mpi_world_size;
    int process_exit_code;
    if (mpi_rank == 0) {
        process_exit_code = master(argc, argv);
    } else {
        process_exit_code = worker(argc, argv);
    }

    MPI_Finalize();

    return process_exit_code;
}