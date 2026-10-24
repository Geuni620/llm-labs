/* Additive training ABI. The checked C/CBLAS kernels remain unchanged. */
#define main frozen_benchmark_main
#include "../cpu-benchmark/experiments/fixedwork_benchmark.c"
#undef main

typedef struct {
    Model model;
    Corpus corpus;
    Sampler sampler;
    uint64_t updates, visited_count;
    unsigned char *visited;
    char *directory;
} NightEngine;

uint32_t kw_abi(void) { return 1; }

void *kw_create(const char *directory,int vocab,int context,int dim,int hidden,int batch,int threads,uint32_t seed) {
    if (!directory || vocab<5 || vocab>65536 || context<1 || context>32 || dim<1 || dim>512 || hidden<1 || hidden>1024
        || batch<1 || batch>1024 || threads<1 || threads>8) fail("invalid training ABI configuration");
    NightEngine *engine=allocate(1,sizeof(*engine));
    engine->directory=strdup(directory);
    if (!engine->directory) fail("directory allocation failed");
    Config c={.vocab=vocab,.context=context,.dim=dim,.hidden=hidden,.batch=batch,.threads=threads,.seed=seed,
              .warmup=0,.steps=1,.corpus_path=engine->directory,
              .kernels={.threads=threads,.softmax_parallel=1,.reduction_parallel=1,.adam_parallel=1,.clip_norm=1.0f}};
    thread_budget(threads);
    engine->corpus=load_corpus(c);
    engine->model=create_model(c);
    engine->sampler=sampler_start(c,engine->corpus.windows);
    engine->visited=allocate(engine->corpus.paragraphs.count,1);
    return engine;
}

void kw_destroy(void *pointer) {
    NightEngine *e=pointer;
    if (!e) return;
    destroy_model(&e->model);free(e->corpus.ids.data);free(e->corpus.paragraphs.data);free(e->corpus.cumulative);
    free(e->visited);free(e->directory);free(e);
}

float *kw_array(void *pointer,int which) {
    NightEngine *e=pointer;
    switch (which) {
        case 0:return e->model.parameters;
        case 1:return e->model.moment1;
        case 2:return e->model.moment2;
        case 3:return e->model.gradients;
        case 4:return e->model.initial;
        default:return NULL;
    }
}
unsigned char *kw_visited(void *pointer) { return ((NightEngine *)pointer)->visited; }

void kw_info(void *pointer,uint64_t *out) {
    NightEngine *e=pointer;
    out[0]=e->model.layout.count;out[1]=e->corpus.windows;out[2]=e->corpus.paragraphs.count;
    out[3]=e->updates;out[4]=e->sampler.consumed;out[5]=e->visited_count;
    out[6]=e->corpus.ids.bytes+e->corpus.paragraphs.bytes+e->corpus.paragraphs.count*sizeof(uint64_t);
    out[7]=e->model.workspace_bytes;
}

void kw_get_sampler(void *pointer,uint64_t *out) {
    Sampler *s=&((NightEngine *)pointer)->sampler;
    out[0]=s->random;out[1]=s->position;out[2]=s->multiplier;out[3]=s->shift;out[4]=s->consumed;
}
int kw_restore_progress(void *pointer,uint64_t updates,const uint64_t *in) {
    NightEngine *e=pointer;
    if (updates>=INT_MAX || in[0]>UINT32_MAX || in[1]>e->corpus.windows || in[2]<1 || in[2]>=e->corpus.windows
        || gcd64(in[2],e->corpus.windows)!=1 || in[3]>=e->corpus.windows
        || (in[1]!=e->corpus.windows && in[4]%e->corpus.windows!=in[1])
        || (in[1]==e->corpus.windows && in[4]%e->corpus.windows!=0)) return 0;
    e->sampler=(Sampler){.random=(uint32_t)in[0],.position=in[1],.multiplier=in[2],.shift=in[3],.consumed=in[4]};
    e->updates=updates;e->visited_count=0;
    for (size_t i=0;i<e->corpus.paragraphs.count;++i) {
        if (e->visited[i]>1) return 0;
        e->visited_count+=e->visited[i]!=0;
    }
    return 1;
}

/* Same affine sampler/window lookup as frozen gather_real, also count visits.
 * The caller caps a batch at an epoch edge, so a saved epoch is exact. */
static void night_gather(NightEngine *e) {
    Model *m=&e->model;Corpus *corpus=&e->corpus;Sampler *sampler=&e->sampler;
    const uint16_t *ids=(const uint16_t *)corpus->ids.data;
    for (int b=0;b<m->config.batch;++b) {
        if (sampler->position==corpus->windows) next_pass(sampler,corpus->windows);
        uint64_t window=(sampler->multiplier*sampler->position+sampler->shift)%corpus->windows;
        ++sampler->position;++sampler->consumed;
        size_t low=0,high=corpus->paragraphs.count;
        while (low<high) { size_t mid=low+(high-low)/2;if(corpus->cumulative[mid]<=window)low=mid+1;else high=mid; }
        uint64_t before=low?corpus->cumulative[low-1]:0;
        uint64_t start=little64(corpus->paragraphs.data+low*28)+window-before;
        for (int j=0;j<m->config.context;++j) m->contexts[(size_t)b*m->config.context+j]=ids[start+j];
        m->targets[b]=ids[start+m->config.context];
        if (!e->visited[low]) { e->visited[low]=1;++e->visited_count; }
    }
}

/* out = loss, norm, scale, actual batch, epoch reached, parameter finite flag.
 * No warmup updates contaminate actual training or its saved state. */
int kw_step(void *pointer,uint64_t limit,double *out) {
    NightEngine *e=pointer;Model *m=&e->model;
    if (e->sampler.consumed>=limit) return 0;
    if (e->updates>=INT_MAX-1) fail("training update counter exceeds supported bound");
    if (e->sampler.position==e->corpus.windows) next_pass(&e->sampler,e->corpus.windows);
    int capacity=m->config.batch;
    uint64_t amount=(uint64_t)capacity;
    if (amount>limit-e->sampler.consumed) amount=limit-e->sampler.consumed;
    if (amount>e->corpus.windows-e->sampler.position) amount=e->corpus.windows-e->sampler.position;
    m->config.batch=(int)amount;
    night_gather(e);
    float loss=update(m,0,(int)e->updates+1);
    ++e->updates;
    out[0]=loss;out[1]=m->gradient_norm;out[2]=m->clipping_scale;out[3]=(double)amount;
    out[4]=(double)e->sampler.consumed/(double)e->corpus.windows;
    m->config.batch=capacity;
    return 1;
}

/* Expose the actual batch for independent stream/numerical verification. */
uint32_t *kw_contexts(void *p) { return ((NightEngine *)p)->model.contexts; }
uint32_t *kw_targets(void *p) { return ((NightEngine *)p)->model.targets; }
int kw_finite(void *p) { return finite_model(&((NightEngine *)p)->model); }

static void night_logits(NightEngine *e,const uint32_t *contexts,int rows) {
    Model *m=&e->model;Config c=m->config;Layout l=m->layout;
    if (rows<1 || rows>c.batch) fail("invalid evaluation batch");
    int cd=c.context*c.dim;
    for (int i=0;i<rows*c.context;++i) {
        if (contexts[i]>=(uint32_t)c.vocab) fail("evaluation ID out of range");
        memcpy(m->input+(size_t)i*c.dim,m->parameters+(size_t)contexts[i]*c.dim,(size_t)c.dim*sizeof(float));
    }
    multiply(m,0,1,rows,c.hidden,cd,1.0f,m->input,cd,m->parameters+l.hidden_weight,cd,0.0f,m->hidden,c.hidden);
    for (int b=0;b<rows;++b) for (int h=0;h<c.hidden;++h) {
        size_t at=(size_t)b*c.hidden+h;m->hidden[at]=tanhf(m->hidden[at]+m->parameters[l.hidden_bias+h]);
    }
    multiply(m,0,1,rows,c.vocab,c.hidden,1.0f,m->hidden,c.hidden,m->parameters+l.output_weight,c.hidden,0.0f,m->logits,c.vocab);
    for (int b=0;b<rows;++b) for (int v=0;v<c.vocab;++v) m->logits[(size_t)b*c.vocab+v]+=m->parameters[l.output_bias+v];
}

void kw_logits(void *p,const uint32_t *contexts,int rows,float *output) {
    NightEngine *e=p;night_logits(e,contexts,rows);
    memcpy(output,e->model.logits,(size_t)rows*e->model.config.vocab*sizeof(float));
}

/* Stable per-target validation NLL; no gradients, moments or sampler changes.
 * Counts/sums are all-target then excluding WS/EOS. Float logits/exp, double sums. */
void kw_evaluate(void *p,const uint32_t *contexts,const uint32_t *targets,int rows,int space_id,double *out) {
    NightEngine *e=p;night_logits(e,contexts,rows);
    int vocab=e->model.config.vocab,threads=e->model.config.threads;
    double values[rows][6];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int b=0;b<rows;++b) {
        if (targets[b]>=(uint32_t)vocab) fail("evaluation target out of range");
        float *row=e->model.logits+(size_t)b*vocab;
        float maximum=-INFINITY;int argmax=0;
        for (int v=0;v<vocab;++v) { if (!finite_float(row[v])) fail("nonfinite evaluation logits");if(row[v]>maximum){maximum=row[v];argmax=v;} }
        float target=row[targets[b]];
        for (int v=0;v<vocab;++v) row[v]-=maximum;
#if EXPERIMENT_VFORCE_EXP
        vvexpf(row,row,&vocab);
#else
        for (int v=0;v<vocab;++v) row[v]=expf(row[v]);
#endif
        double sum=0;for (int v=0;v<vocab;++v) sum+=row[v];
        double loss=log(sum)+(double)maximum-(double)target;
        if (!isfinite(loss) || loss < -1e-8) fail("nonfinite validation loss");
        int content=(int)targets[b]!=space_id && targets[b]!=3;
        values[b][0]=1;values[b][1]=loss;values[b][2]=(uint32_t)argmax==targets[b];
        values[b][3]=content;values[b][4]=content?loss:0;values[b][5]=content?values[b][2]:0;
    }
    for (int k=0;k<6;++k) {out[k]=0;for(int b=0;b<rows;++b)out[k]+=values[b][k];}
}
