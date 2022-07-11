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
    // we use array of chars to store number
    // as a string.
    char str[10];
    // Print in string the value of var
    // with two decimal point
    sprintf(str, "%.3f", var);
    // scan string value in var
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
            // TODO: use compound_times
            return pow(1.0 / df, 1.0 / day_count) - 1.0;
        case Convention::EXPONENTIAL:
            return log(df) * (1.0 / day_count);
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

std::vector<double> get_discount_factors_1_T(int years, double r, int compound_times = 1, Convention convention = Convention::YIELD)
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

std::vector<double> get_discount_factors_0_T_less1(int years, double r, int compound_times = 1, Convention convention = Convention::YIELD)
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

double annuity_npv(double coupon, int maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
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

double coupon_npv(double npv, int maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
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
double bond_npv(double face_value, double coupon_rate, int maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
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

double bond_fv(double coupon, int maturity, double interest_rate, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double fv = 0.0;
    auto dfs = get_discount_factors_0_T_less1(maturity, interest_rate, compound_times, convention);
    for(const auto& df : dfs)
    {
        fv += (1.0 / df);
    }
    return coupon * fv;
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

    std::cout << equivalent_rate(0.05, 1, 2) << std::endl;

    return 0;
}
