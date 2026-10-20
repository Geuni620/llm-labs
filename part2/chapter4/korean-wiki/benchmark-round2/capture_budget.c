/* Verification-only wrapper around the unchanged generic C engine. */
#define main frozen_benchmark_main
#include "../cpu-benchmark/experiments/fixedwork_benchmark.c"
#undef main

static const char *capture_directory;
static void save_capture(const char *name, const void *data, size_t count, size_t item_size) {
    char path[4096];
    if (snprintf(path,sizeof(path),"%s/%s.bin",capture_directory,name)>=(int)sizeof(path)) fail("capture path too long");
    FILE *stream=fopen(path,"wb");
    if (!stream || fwrite(data,item_size,count,stream)!=count || fclose(stream)) fail("capture write failed");
}
static void save_step(const Model *model,int step,const char *field,const void *values,size_t count,size_t size) {
    (void)model;
    char name[128];
    snprintf(name,sizeof(name),"step%d.%s",step,field);
    save_capture(name,values,count,size);
}
int main(int argc,char **argv) {
    capture_directory=getenv("KOWIKI_CAPTURE_DIRECTORY");
    if (!capture_directory) fail("capture directory required");
    Config c=arguments(argc,argv);
    if (!c.corpus_path || c.batch!=512 || (c.hidden!=64 && c.hidden!=128)) fail("unexpected capture configuration");
    c.steps=4; c.warmup=2; c.poison_gradients=1;
    thread_budget(c.threads);
    Corpus corpus=load_corpus(c);
    Model model=create_model(c);
    model.check_raw_gradients=allocate(model.layout.count,sizeof(float));
    Sampler sampler=sampler_start(c,corpus.windows);
    save_capture("initial",model.initial,model.layout.count,sizeof(float));
    for (int step=1;step<=2;++step) { gather_real(&model,&corpus,&sampler,0); update(&model,0,step); }
    memcpy(model.parameters,model.initial,model.layout.count*sizeof(float));
    memset(model.moment1,0,model.layout.count*sizeof(float));
    memset(model.moment2,0,model.layout.count*sizeof(float));
    memset(model.gradients,0,model.layout.count*sizeof(float));
    memset(model.norm_scratch,0,model.norm_capacity*sizeof(double));
    sampler=sampler_start(c,corpus.windows);
    save_capture("reset_params",model.parameters,model.layout.count,sizeof(float));
    save_capture("reset_m1",model.moment1,model.layout.count,sizeof(float));
    save_capture("reset_m2",model.moment2,model.layout.count,sizeof(float));
    printf("{\"config\":");print_config(c);
    printf(",\"parameter_count\":%zu,\"initial_checksum\":\"%016" PRIx64 "\",\"steps\":[",model.layout.count,model.initial_checksum);
    for (int step=1;step<=4;++step) {
        gather_real(&model,&corpus,&sampler,0);
        save_step(&model,step,"contexts",model.contexts,(size_t)c.batch*c.context,sizeof(uint32_t));
        save_step(&model,step,"targets",model.targets,(size_t)c.batch,sizeof(uint32_t));
        float loss=update(&model,0,step);
        if (!finite_model(&model)) fail("nonfinite captured model");
        save_step(&model,step,"raw_gradients",model.check_raw_gradients,model.layout.count,sizeof(float));
        save_step(&model,step,"clipped_gradients",model.gradients,model.layout.count,sizeof(float));
        save_step(&model,step,"updated_params",model.parameters,model.layout.count,sizeof(float));
        save_step(&model,step,"moment1",model.moment1,model.layout.count,sizeof(float));
        save_step(&model,step,"moment2",model.moment2,model.layout.count,sizeof(float));
        printf("%s{\"step\":%d,\"loss\":%.9g,\"gradient_norm\":%.9g,\"clipping_scale\":%.9g,\"batch_checksum\":\"%016" PRIx64 "\"}",step>1?",":"",step,loss,model.gradient_norm,model.clipping_scale,batch_checksum(&model,0));
    }
    printf("],\"gradient_nan_poison\":true,\"sampler_final\":");print_sampler(&sampler);printf("}\n");
    destroy_model(&model);free(corpus.ids.data);free(corpus.paragraphs.data);free(corpus.cumulative);
    return 0;
}
