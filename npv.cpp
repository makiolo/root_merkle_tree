#include <iostream>
#include <cmath>
#include <vector>

enum Convention
{
    LINEAR,
    YIELD,
    EXPONENTIAL,
};

double round3(double var)
{
    char str[10];
    sprintf(str, "%.3f", var);
    var = strtof(str, nullptr);
    return var;
}

double df2zc(double df, double day_count, int compound_times=1, Convention conv = Convention::YIELD)
{
    switch(conv)
    {
        case Convention::LINEAR:
            return (1.0 / df - 1.0) * (1.0 / day_count);
        case Convention::YIELD:
            return (pow(1.0 / df, 1.0 / (day_count * compound_times)) - 1.0) * compound_times;
        case Convention::EXPONENTIAL:
            return -log(df) / day_count;
        default:
            throw std::runtime_error("Invalid convention");
    }
}

double zc2df(double zc, double day_count, int compound_times=1, Convention conv = Convention::YIELD)
{
    switch(conv)
    {
        case Convention::LINEAR:
            return 1.0 / (1.0 + zc * day_count);
        case Convention::YIELD:
            return 1.0 / (pow((1.0 + zc / compound_times), day_count * compound_times));
        case Convention::EXPONENTIAL:
            return exp(-zc * day_count);
        default:
            throw std::runtime_error("Invalid convention");
    }
}

double equivalent_rate(double rate, int compound_times, int other_compound_times=1)
{
    return other_compound_times * pow(1.0 + (rate / compound_times), double(compound_times) / other_compound_times) - other_compound_times;
}

std::vector<double> get_discount_factors_1_T(double years, double r, int compound_times = 1, Convention convention = Convention::YIELD)
{
    std::vector<double> dfs;
    for(int i=1; i <= years * compound_times; ++i)
    {
        // Z(x) = Z(i / compound_times)
        double df = zc2df(r, double(i) / compound_times, compound_times, convention);
        // df = round3(df);
        dfs.push_back(df);
    }
    return dfs;
}

std::vector<double> get_discount_factors_0_T_less1(double years, double r, int compound_times = 1, Convention convention = Convention::YIELD)
{
    std::vector<double> dfs;
    for(int i=0; i < years * compound_times; ++i)
    {
        // Z(x) = Z(i / compound_times)
        double df = zc2df(r, double(i) / compound_times, compound_times, convention);
        dfs.push_back(df);
    }
    return dfs;
}

double annuity_npv(double coupon, double maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
{
    auto dfs = get_discount_factors_1_T(maturity, interest_rate, compound_times, convention);
    double total_df = 0.0;
    for(const auto& df : dfs)
    {
        total_df += df;
    }
    double npv = total_df*(coupon / compound_times);
    return npv;
}

double coupon_npv(double npv, double maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
{
    auto dfs = get_discount_factors_1_T(maturity, interest_rate, compound_times, convention);
    double total_df = 0.0;
    for(const auto& df : dfs)
    {
        total_df += df;
    }
    // cuota anual
    return (npv * compound_times) / total_df;
    // cuota periodo
    // return npv / total_df;
}

// from interest rate (no from discount factor)
double bond_npv(double face_value, double coupon_rate, double maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupon = coupon_rate * face_value;
    auto dfs = get_discount_factors_1_T(maturity, interest_rate, compound_times, convention);
    double total_df = 0.0;
    for(const auto& df : dfs)
    {
        total_df += df;
    }
    auto n = dfs.size();
    double npv = total_df*(coupon / compound_times) + dfs[n - 1] * face_value;
    return npv;
}

double invest_npv(double investment, double coupon_rate, double maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupon = coupon_rate * investment;
    auto dfs = get_discount_factors_1_T(maturity, interest_rate, compound_times, convention);
    double total_df = 0.0;
    for(const auto& df : dfs)
    {
        total_df += df;
    }
    auto n = dfs.size();
    double npv = total_df*(coupon / compound_times) - investment;
    return npv;
}

double stock_npv(double investment, double coupon_rate, double maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupon = coupon_rate * investment;
    auto dfs = get_discount_factors_1_T(maturity, interest_rate, compound_times, convention);
    double total_df = 0.0;
    for(const auto& df : dfs)
    {
        total_df += df;
    }
    auto n = dfs.size();
    double npv = total_df*(coupon / compound_times) - investment + dfs[n - 1] * investment;;
    return npv;
}

double bond_fv(double coupon, double maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double fv = 0.0;
    auto dfs = get_discount_factors_0_T_less1(maturity, interest_rate, compound_times, convention);
    for(const auto& df : dfs)
    {
        fv += (1.0 / df);
    }
    return coupon * fv;
}

double interest_on_capital(double initial, double final, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double r = (final - initial) / initial;
    double df = zc2df(r, 1, 1, Convention::LINEAR);  // 1.0 / (1.0 + zc * day_count);
    return df2zc(df, maturity, compound_times, convention);  // -log(df) / day_count   etc ...
}

double cagr(double initial, double final, double maturity)
{
    return pow(final / initial, 1.0 / maturity) - 1.0;
}

double future_value(double initial, double r, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double df = zc2df(r, maturity, compound_times, convention);
    return initial / df;
}

int main() {

    // valor presente de un bono
    // valorar un bono que da un yield "seguro" haciendo otros proyectos risk free
    std::cout << bond_npv(15,
                          // inversion "segura" ofrecida por el bono
                          0.16, 3,
                          // inversion libre de riesgo en el mercado
                          0.03, 1, Convention::YIELD) << std::endl;

    // valor futuro
    // Ejemplo: cuanto dinero tendremos en el banco ahorrado en 30 años ?
    std::cout << bond_fv(8000, 5,
                          // inversion libre de riesgo en el mercado
                         0.03, 1, Convention::YIELD) << std::endl;

    // valor presente de las anualidades
    std::cout << annuity_npv(650*12.0, 5, 0.065, 1, Convention::YIELD) << std::endl;

    std::cout << coupon_npv(180000, 40, 0.03, 1, Convention::YIELD) / 12.0 << std::endl;

    std::cout << zc2df(df2zc(0.95, 3, 4, Convention::EXPONENTIAL), 3, 4, Convention::EXPONENTIAL) << std::endl;
    std::cout << df2zc(zc2df(0.05, 3, 4, Convention::EXPONENTIAL), 3, 4, Convention::EXPONENTIAL) << std::endl;
    std::cout << "--> " << equivalent_rate(0.10, 12, 1) << std::endl;
    std::cout << zc2df(0.05, 3, 4, Convention::YIELD) << std::endl;
    std::cout << zc2df(0.0509453, 3, 1, Convention::YIELD) << std::endl;
    std::cout << df2zc(0.861509, 3, 4, Convention::YIELD) << std::endl;
    std::cout << df2zc(0.861509, 3) << std::endl;

    //                      inversion inicial
    std::cout << invest_npv(6000,
                          // cuota% y años
                          0.2, 3,
                          // free risk rate y tipo de porcentaje
                          0.08, 1, Convention::YIELD) << std::endl;
                          
    //                      inversion inicial en el stock
    std::cout << stock_npv(10000,
                          // cuota% y años que venderás
                          0.2, 10,
                          // dividendo y tipo de porcentaje
                          0.03, 1, Convention::YIELD) << std::endl;
                          
    double initial = 10000;
    double final = 20000;
    double past_years = 1;
    double forward_years = 1;
    // past info
    double r = interest_on_capital(initial, final, past_years, 12, Convention::YIELD);
    std::cout << r * 100.0 << std::endl;
    // forward prediction
    std::cout << future_value(final, r, forward_years, 12, Convention::YIELD) << std::endl;
    
    // trading
    initial = 5000;
    r = 0.10;
    std::cout << "mensual " << r*100 << std::endl;
    forward_years = 3.0;
    std::cout << future_value(initial, r, forward_years, 12, Convention::YIELD) << std::endl;
    
    double r2 = equivalent_rate(r, 12, 1);
    std::cout << "anual " << r2*100 << std::endl;
    std::cout << future_value(initial, r2, forward_years, 1, Convention::YIELD) << std::endl;

    return 0;
}
