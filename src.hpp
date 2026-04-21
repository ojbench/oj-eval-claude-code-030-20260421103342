#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

// Problem requires:
// typedef std::vector<std::vector<double> > IMAGE_T;
// int judge(IMAGE_T &img);

typedef std::vector<std::vector<double> > IMAGE_T;

namespace _nr_internal {
    static inline int clampi(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }

    // Compute Otsu threshold (values in [0,1]) returning threshold in [0,1]
    static inline double otsu_threshold(const IMAGE_T &img){
        const int H = (int)img.size();
        const int W = H ? (int)img[0].size() : 0;
        if (H == 0 || W == 0) return 0.5; // fallback
        const int bins = 256;
        std::vector<double> hist(bins, 0.0);
        double total = 0.0;
        for(int y=0;y<H;y++){
            for(int x=0;x<W;x++){
                double v = img[y][x];
                if (v < 0.0) v = 0.0; if (v > 1.0) v = 1.0;
                int b = (int)std::floor(v * (bins-1));
                hist[b] += 1.0;
                total += 1.0;
            }
        }
        if (total <= 0.0) return 0.5;
        // Normalize
        for(int i=0;i<bins;i++) hist[i] /= total;
        std::vector<double> w(bins, 0.0), mu(bins, 0.0);
        w[0] = hist[0]; mu[0] = 0.0 * hist[0];
        for(int t=1;t<bins;t++){
            w[t] = w[t-1] + hist[t];
            mu[t] = mu[t-1] + (double)t * hist[t];
        }
        double muT = mu[bins-1];
        double max_sigma = -1.0; int best_t = bins/2;
        for(int t=0;t<bins;t++){
            double w0 = w[t];
            double w1 = 1.0 - w0;
            if (w0 <= 1e-9 || w1 <= 1e-9) continue;
            double mu0 = mu[t] / w0;
            double mu1 = (muT - mu[t]) / w1;
            double diff = (mu0 - mu1);
            double sigma_b = w0 * w1 * diff * diff;
            if (sigma_b > max_sigma){ max_sigma = sigma_b; best_t = t; }
        }
        return (double)best_t / (double)(bins-1);
    }

    static inline void binarize(const IMAGE_T &img, std::vector< std::vector<unsigned char> > &bin){
        const int H = (int)img.size();
        const int W = H ? (int)img[0].size() : 0;
        bin.assign(H, std::vector<unsigned char>(W, 0));
        if (H == 0 || W == 0) return;
        double T = otsu_threshold(img);
        // Ensure foreground is white (value 1). According to problem, digits are white, background black.
        for(int y=0;y<H;y++){
            for(int x=0;x<W;x++){
                double v = img[y][x];
                bin[y][x] = (unsigned char)((v >= T) ? 1 : 0);
            }
        }
        // If foreground seems too sparse or too dense, adjust threshold slightly
        int cnt = 0;
        for(int y=0;y<H;y++) for(int x=0;x<W;x++) cnt += bin[y][x] ? 1 : 0;
        double ratio = (H*W) ? (double)cnt / (double)(H*W) : 0.0;
        if (ratio < 0.02 || ratio > 0.5){
            // Use mean as fallback
            double sum = 0.0;
            for(int y=0;y<H;y++) for(int x=0;x<W;x++) sum += img[y][x];
            double mean = (H*W) ? sum/(H*W) : 0.5;
            for(int y=0;y<H;y++){
                for(int x=0;x<W;x++){
                    bin[y][x] = (unsigned char)((img[y][x] >= mean) ? 1 : 0);
                }
            }
        }
    }

    struct BBox{
        int x0, y0, x1, y1; // inclusive bounds
        BBox(): x0(0), y0(0), x1(-1), y1(-1) {}
        BBox(int _x0,int _y0,int _x1,int _y1): x0(_x0), y0(_y0), x1(_x1), y1(_y1) {}
        int width() const { return x1 - x0 + 1; }
        int height() const { return y1 - y0 + 1; }
        bool valid() const { return x0 <= x1 && y0 <= y1; }
    };

    static inline BBox bounding_box(const std::vector< std::vector<unsigned char> > &bin){
        const int H = (int)bin.size();
        const int W = H ? (int)bin[0].size() : 0;
        int x0=W, y0=H, x1=-1, y1=-1;
        for(int y=0;y<H;y++){
            for(int x=0;x<W;x++) if (bin[y][x]){
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
        }
        if (x1 < x0 || y1 < y0) return BBox(0,0,-1,-1);
        return BBox(x0,y0,x1,y1);
    }

    static inline int count_holes(const std::vector< std::vector<unsigned char> > &bin, const BBox &b){
        if (!b.valid()) return 0;
        int W = (int)bin[0].size();
        int H = (int)bin.size();
        std::vector< std::vector<unsigned char> > vis(H, std::vector<unsigned char>(W, 0));
        int holes = 0;
        for(int y=b.y0;y<=b.y1;y++){
            for(int x=b.x0;x<=b.x1;x++){
                if (bin[y][x] == 0 && !vis[y][x]){
                    std::queue< std::pair<int,int> > q; q.push(std::make_pair(x,y)); vis[y][x]=1;
                    bool touchBorder = (x==b.x0||x==b.x1||y==b.y0||y==b.y1);
                    while(!q.empty()){
                        std::pair<int,int> p = q.front(); q.pop();
                        int cx = p.first, cy = p.second;
                        static const int dx[4]={1,-1,0,0};
                        static const int dy[4]={0,0,1,-1};
                        for(int k=0;k<4;k++){
                            int nx=cx+dx[k], ny=cy+dy[k];
                            if (!(ny>=b.y0 && ny<=b.y1 && nx>=b.x0 && nx<=b.x1)) continue;
                            if (!vis[ny][nx] && bin[ny][nx]==0){
                                vis[ny][nx]=1; q.push(std::make_pair(nx,ny));
                                if (nx==b.x0||nx==b.x1||ny==b.y0||ny==b.y1) touchBorder = true;
                            }
                        }
                    }
                    if (!touchBorder) holes++;
                }
            }
        }
        return holes;
    }

    struct Feat{
        double fg_ratio; // foreground / bbox area
        double top_ratio, bottom_ratio;
        double left_ratio, right_ratio;
        double mid_row_ratio, top_band_ratio, bottom_band_ratio;
        double qUL, qUR, qLL, qLR;
        double cx, cy; // center of mass normalized [0,1] within bbox
        int col_transitions; // transitions along center column
        int row_transitions; // transitions along center row
        int width, height;
        int holes;
        Feat(): fg_ratio(0), top_ratio(0), bottom_ratio(0), left_ratio(0), right_ratio(0),
                mid_row_ratio(0), top_band_ratio(0), bottom_band_ratio(0),
                qUL(0), qUR(0), qLL(0), qLR(0), cx(0.5), cy(0.5),
                col_transitions(0), row_transitions(0), width(0), height(0), holes(0) {}
    };

    static inline Feat extract_features(const std::vector< std::vector<unsigned char> > &bin, const BBox &b){
        Feat f; f.width = b.width(); f.height = b.height();
        if (!b.valid()) return f;
        int area = f.width * f.height;
        int fg=0;
        int top=0,bottom=0,left=0,right=0;
        int midrow=0, topband=0, bottomband=0;
        int ul=0, ur=0, ll=0, lr=0;
        double sumx=0.0, sumy=0.0;
        int mid_y = b.y0 + f.height/2;
        int mid_x = b.x0 + f.width/2;
        int colTrans=0, rowTrans=0;
        // Count transitions along center column and row
        int prev = -1;
        for(int y=b.y0;y<=b.y1;y++){
            int v = (bin[y][mid_x] ? 1 : 0);
            if (prev==-1) prev = v; else if (v!=prev){ colTrans++; prev=v; }
        }
        prev = -1;
        for(int x=b.x0;x<=b.x1;x++){
            int v = (bin[mid_y][x] ? 1 : 0);
            if (prev==-1) prev = v; else if (v!=prev){ rowTrans++; prev=v; }
        }
        for(int y=b.y0;y<=b.y1;y++){
            for(int x=b.x0;x<=b.x1;x++) if (bin[y][x]){
                fg++;
                sumx += (x - b.x0 + 0.5);
                sumy += (y - b.y0 + 0.5);
                if (y < b.y0 + f.height/2) top++; else bottom++;
                if (x < b.x0 + f.width/2) left++; else right++;
                if (y == mid_y) midrow++;
                if (y < b.y0 + (int)std::ceil(f.height*0.25)) topband++; else if (y >= b.y1 - (int)std::floor(f.height*0.25)) bottomband++;
                if (x < b.x0 + f.width/2 && y < b.y0 + f.height/2) ul++;
                else if (x >= b.x0 + f.width/2 && y < b.y0 + f.height/2) ur++;
                else if (x < b.x0 + f.width/2 && y >= b.y0 + f.height/2) ll++;
                else lr++;
            }
        }
        f.fg_ratio = area ? (double)fg / (double)area : 0.0;
        f.top_ratio = area ? (double)top / (double)area : 0.0;
        f.bottom_ratio = area ? (double)bottom / (double)area : 0.0;
        f.left_ratio = area ? (double)left / (double)area : 0.0;
        f.right_ratio = area ? (double)right / (double)area : 0.0;
        f.mid_row_ratio = area ? (double)midrow / (double)area : 0.0;
        f.top_band_ratio = area ? (double)topband / (double)area : 0.0;
        f.bottom_band_ratio = area ? (double)bottomband / (double)area : 0.0;
        f.qUL = area ? (double)ul / (double)area : 0.0;
        f.qUR = area ? (double)ur / (double)area : 0.0;
        f.qLL = area ? (double)ll / (double)area : 0.0;
        f.qLR = area ? (double)lr / (double)area : 0.0;
        f.cx = (fg>0) ? (sumx / (double)fg) / (double)f.width : 0.5;
        f.cy = (fg>0) ? (sumy / (double)fg) / (double)f.height : 0.5;
        f.col_transitions = colTrans;
        f.row_transitions = rowTrans;
        return f;
    }

    static inline int classify(const std::vector< std::vector<unsigned char> > &bin){
        // Compute bbox and features
        BBox b = bounding_box(bin);
        if (!b.valid()) return 0; // fallback
        Feat f = extract_features(bin, b);
        // Count holes
        int holes = count_holes(bin, b);
        f.holes = holes;

        double ar = (f.height>0) ? (double)f.width / (double)f.height : 1.0;
        double top_minus_bottom = f.top_ratio - f.bottom_ratio;
        double right_minus_left = f.right_ratio - f.left_ratio;

        // Rules
        if (holes >= 2) return 8;
        if (holes == 1){
            if (top_minus_bottom > 0.08) return 9;
            if (-top_minus_bottom > 0.08) return 6;
            // Symmetric -> likely 0
            return 0;
        }

        // Narrow tall: 1/7/4
        if (ar < 0.60){
            if (f.top_band_ratio > 0.45 && right_minus_left > 0.05) return 7;
            if (f.col_transitions >= 2 && f.mid_row_ratio > 0.035 && f.top_ratio > f.bottom_ratio) return 4;
            return 1;
        }

        // Strongly top-heavy and right-heavy -> 7
        if (f.top_band_ratio > 0.48 && f.bottom_band_ratio < 0.25 && right_minus_left > 0.08) return 7;

        // 4: two vertical segments through center and strong mid bar
        if (f.col_transitions >= 2 && f.mid_row_ratio > 0.04 && f.top_ratio > f.bottom_ratio) return 4;

        // 3: right-heavy and two lobes
        if (f.col_transitions >= 2 && right_minus_left > 0.12) return 3;

        // 2: top-heavy, right-biased, bottom-left light, bottom-right present
        if (f.top_ratio > f.bottom_ratio && right_minus_left > 0.05 && f.qLR > f.qLL + 0.02) return 2;

        // 5: top-heavy, left-biased, bottom-right light
        if (f.top_ratio > f.bottom_ratio && right_minus_left < -0.03 && f.qLR < 0.06) return 5;

        // 9 vs 6 fallback
        if (top_minus_bottom > 0.15) return 9;
        if (-top_minus_bottom > 0.15) return 6;

        // Center transitions 1 -> likely 0 or 1; choose 0 here
        if (f.col_transitions <= 1 && f.row_transitions <= 1) return 0;

        // Fallback guesses based on quadrant balance
        if (right_minus_left > 0.0) return 3; // lean to right -> 3
        if (right_minus_left < 0.0) return 5; // lean to left -> 5
        return 0;
    }
}

int judge(IMAGE_T &img){
    // Binarize
    std::vector< std::vector<unsigned char> > bin;
    _nr_internal::binarize(img, bin);
    // Classify
    int pred = _nr_internal::classify(bin);
    // Ensure valid return
    if (pred < 0) pred = 0; if (pred > 9) pred = 9;
    return pred;
}
