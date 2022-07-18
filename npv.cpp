// https://www.linkedin.com/pulse/python-bootstrapping-zero-curve-sheikh-pancham#:~:text=The%20objective%20of%20bootstrapping%20is,the%20end%20of%20its%20term.

#include <iostream>
#include <cmath>
#include <vector>
#include <catch_amalgamated.hpp>

class InterestRate;
class DiscountFactor;

enum Convention
{
    LINEAR,
    YIELD,
    EXPONENTIAL,
};

// convert df and zc
double df2zc(double df, double maturity, int compound_times = 1, Convention conv = Convention::YIELD);
double zc2df(double zc, double maturity, int compound_times = 1, Convention conv = Convention::YIELD);
InterestRate equivalent_rate(double rate, int compound_times, Convention convention = Convention::YIELD, int other_compound_times = 1, Convention other_convention = Convention::YIELD);
InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times = 1);
std::vector<DiscountFactor> get_discount_factors_1_T(const InterestRate& ir, double years);
std::vector<DiscountFactor> get_discount_factors_0_T_less1(const InterestRate& ir, double years);
std::vector<DiscountFactor> get_discount_factors_1_T(double r, double years, int compound_times = 1, Convention convention = Convention::YIELD);
std::vector<DiscountFactor> get_discount_factors_0_T_less1(double r, double years, int compound_times = 1, Convention convention = Convention::YIELD);

// one cash flow
double to_present_value(double cash, const InterestRate& r, double maturity);
double to_future_value(double cash, const InterestRate& r, double maturity);

// real
double real_from_coupon(double coupon, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);
double coupon_from_real(double real, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);

// fv - coupon - fv
double coupon_from_npv(double npv, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);
double coupon_from_fv(double fv, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);
double npv_from_coupon(double coupon, const InterestRate& interest_rate, double years);
double npv_from_coupon(double coupon, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);
double fv_from_coupon(double coupon, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);

// growth_coupon
double npv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD, int g_compound_times = 1, Convention g_convention = Convention::YIELD);
double fv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD, int g_compound_times = 1, Convention g_convention = Convention::YIELD);
double coupon_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD, int g_compound_times = 1, Convention g_convention = Convention::YIELD);

// value products
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, double maturity);
double bond_npv(double face_value, double coupon, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);
double stock_npv(double investment, double dividend, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);
double stock_fv(double investment, double dividend, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);
double stock_real(double investment, double dividend, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);

// calcular r
InterestRate on_capital(double initial, double final, double maturity = 1.0, Convention convention = Convention::YIELD, int compound_times = 1);

// deprecated
double round3(double var);

class DiscountFactor
{
public:
    DiscountFactor(double df_)
        : df(df_)
    {
        
    }

    DiscountFactor(const DiscountFactor& other)
    {
        df = other.df;
    }

    DiscountFactor(DiscountFactor&& other)
    {
        df = std::move(other.df);
    }

    InterestRate to_interest_rate(double maturity, Convention convention_ = Convention::YIELD, int compound_times_ = 1) const;

public:
    double df;
};

class InterestRate
{
public:
    InterestRate(double interest_rate_, Convention convention_ = Convention::YIELD, int compound_times_ = 1)
        : r(interest_rate_)
        , conv(convention_)
        , c(compound_times_)
    {

    }

    InterestRate(const InterestRate& other)
    {
        r = other.r;
        c = other.c;
        conv = other.conv;
    }

    InterestRate(InterestRate&& other)
    {
        r = std::move(other.r);
        c = std::move(other.c);
        conv = std::move(other.conv);
    }

    bool operator==(const InterestRate& rhs) const
    {
        return  r == rhs.r &&
                c == rhs.c &&
                conv == rhs.conv;
    }

    DiscountFactor to_discount_factor(double maturity) const
    {
        return DiscountFactor(zc2df(r, maturity, c, conv));
    }

    InterestRate to_other_interest_rate(Convention other_convention, int other_compound_times = 1) const
    {
        return equivalent_rate(r, c, conv, other_compound_times, other_convention);
    }

public:
    double r;  // annual rate
    int c;  // reinversions each year
    Convention conv;  // convention
};

// TODO:
class CashFlow
{
public:
    CashFlow(double maturity_, double cash_)
        : maturity(maturity_)
        , cash(cash_)
    {
        
    }

public:
    double cash;
    double maturity;
};

// TODO:
class Coupon
{
public:
    Coupon(double years_, double cash_)
        : years(years_)
        , cash(cash_)
    {

    }
public:
    double cash;
    double years;
};

InterestRate DiscountFactor::to_interest_rate(double maturity, Convention convention_, int compound_times_) const
{
    return InterestRate(df2zc(df, maturity, compound_times_, convention_), convention_, compound_times_);
}

// ********************** //

/*
double round3(double var)
{
    char str[10];
    sprintf(str, "%.3f", var);
    var = strtof(str, nullptr);
    return var;
}
*/

double df2zc(double df, double maturity, int compound_times, Convention conv)
{
    switch (conv)
    {
        case Convention::LINEAR:
            return (1.0 / df - 1.0) * (1.0 / maturity);
        case Convention::YIELD:
            return (pow(1.0 / df, 1.0 / (maturity * compound_times)) - 1.0) * compound_times;
        case Convention::EXPONENTIAL:
            return -log(df) / maturity;
        default:
            throw std::runtime_error("Invalid convention");
    }
}

double zc2df(double zc, double maturity, int compound_times, Convention conv)
{
    switch (conv)
    {
        case Convention::LINEAR:
            return 1.0 / (1.0 + zc * maturity);
        case Convention::YIELD:
            return 1.0 / (pow((1.0 + zc / compound_times), maturity * compound_times));
        case Convention::EXPONENTIAL:
            return exp(-zc * maturity);
        default:
            throw std::runtime_error("Invalid convention");
    }
}

InterestRate equivalent_rate(double rate, int compound_times, Convention convention, int other_compound_times, Convention other_convention)
{
    return InterestRate(rate, convention, compound_times)
        .to_discount_factor(1.0)
        .to_interest_rate(1.0, other_convention, other_compound_times);
}

InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times)
{
    return equivalent_rate(rate, compound_times, Convention::YIELD, other_compound_times, Convention::YIELD);
}

std::vector<DiscountFactor> get_discount_factors_1_T(const InterestRate& ir, double years)
{
    std::vector<DiscountFactor> dfs;
    for (int i = 1; i <= years * ir.c; ++i)
    {
        dfs.push_back(ir.to_discount_factor(double(i) / ir.c));
    }
    return dfs;
}

std::vector<DiscountFactor> get_discount_factors_0_T_less1(const InterestRate& ir, double years)
{
    std::vector<DiscountFactor> dfs;
    for (int i = 0; i < years * ir.c; ++i)
    {
        dfs.push_back(ir.to_discount_factor(double(i) / ir.c));
    }
    return dfs;
}

std::vector<DiscountFactor> get_discount_factors_1_T(double r, double years, int compound_times, Convention convention)
{
    std::vector<DiscountFactor> dfs;
    InterestRate ir(r, convention, compound_times);
    for (int i = 1; i <= years * compound_times; ++i)
    {
        dfs.push_back(ir.to_discount_factor(double(i) / compound_times));
    }
    return dfs;
}

std::vector<DiscountFactor> get_discount_factors_0_T_less1(double r, double years, int compound_times, Convention convention)
{
    std::vector<DiscountFactor> dfs;
    InterestRate ir(r, convention, compound_times);
    for (int i = 0; i < years * compound_times; ++i)
    {
        dfs.push_back(ir.to_discount_factor(double(i) / compound_times));
    }
    return dfs;
}

// tenemos un cash en "maturity" y nos lo traemos a presente
double to_present_value(double cash, const InterestRate& r, double maturity)
{
    return cash * r.to_discount_factor(maturity).df;
}

// tenemos un cash en "maturity" y nos lo traemos a futuro 
double to_future_value(double cash, const InterestRate& r, double maturity)
{
    return cash / r.to_discount_factor(maturity).df;
}

double real_from_coupon(double coupon, double maturity, int compound_times, Convention convention)
{
    return coupon * maturity * compound_times;
}

double coupon_from_real(double real, double maturity, int compound_times, Convention convention)
{
    return real / (maturity * compound_times);
}

// only coupons
double npv_from_coupon(double coupon, const InterestRate& interest_rate, double years)
{
    auto dfs = get_discount_factors_1_T(interest_rate, years);
    double npv = 0.0;
    for (const auto& df : dfs)
    {
        npv += df.df * (coupon / interest_rate.c);
    }
    return npv;
}

// only coupons
double npv_from_coupon(double coupon, double interest_rate, double maturity, int compound_times, Convention convention)
{
    auto dfs = get_discount_factors_1_T(interest_rate, maturity, compound_times, convention);
    double npv = 0.0;
    for (const auto& df : dfs)
    {
        npv += df.df * (coupon / compound_times);
    }
    return npv;
}

double npv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times, Convention convention, int g_compound_times, Convention g_convention)
{
    auto dfs = get_discount_factors_1_T(interest_rate, maturity, compound_times, convention);
    double npv = 0.0;
    double i = 0.0;
    for (const auto& df : dfs)
    {
        npv += (df.df * (coupon / compound_times)) / zc2df(g, i, g_compound_times, g_convention);
        i += 1.0;
    }
    return npv;
}

double coupon_from_npv(double npv, double interest_rate, double maturity, int compound_times, Convention convention)
{
    auto dfs = get_discount_factors_1_T(interest_rate, maturity, compound_times, convention);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += df.df;
    }
    return (npv * compound_times) / total_df;
}

// VAN = coupons - initial investement
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, double maturity)
{
    double coupons = npv_from_coupon(coupon, interest_rate, maturity);
    double npv = coupons - investment;
    return npv;
}

// coupons + payment on yield-maturity
double bond_npv(double face_value, double coupon, double interest_rate, double maturity, int compound_times, Convention convention)
{
    double coupons = npv_from_coupon(coupon, interest_rate, maturity, compound_times, convention);
    double yield_on_payment = to_present_value(face_value, InterestRate(interest_rate, convention, compound_times), maturity);
    return coupons + yield_on_payment;
}

// stock investment = coupons + payment on yield-maturity - initial investement
double stock_npv(double investment, double dividend, double interest_rate, double maturity, int compound_times, Convention convention)
{
    double coupon = investment * (dividend - interest_rate);
    double coupons = npv_from_coupon(coupon, interest_rate, maturity, compound_times, convention);
    double yield_on_payment = to_present_value(investment, InterestRate(interest_rate, convention, compound_times), maturity);
    double npv = coupons + yield_on_payment - investment;
    return npv;
}

double stock_fv(double investment, double dividend, double interest_rate, double maturity, int compound_times, Convention convention)
{
    double coupon = investment * (dividend - interest_rate);
    //double investment_fv = to_future_value(investment, interest_rate, maturity, compound_times, convention);
    double coupons = fv_from_coupon(coupon, interest_rate, maturity, compound_times, convention);
    //double yield_on_payment = to_future_value(investment, interest_rate, maturity, compound_times, convention);
    //double fv = coupons + yield_on_payment - investment_fv;
    double npv = coupons;
    return npv;
}

double stock_real(double investment, double dividend, double interest_rate, double maturity, int compound_times, Convention convention)
{
    double coupon = investment * (dividend - interest_rate);
    double investment_fv = to_future_value(investment, InterestRate(interest_rate, convention, compound_times), maturity);
    double coupons = real_from_coupon(coupon, maturity, compound_times, convention);
    //double yield_on_payment = to_future_value(investment, interest_rate, maturity, compound_times, convention);
    //double fv = coupons + yield_on_payment - investment_fv;
    double npv = investment - coupons - investment_fv;
    return npv;
}

double fv_from_coupon(double coupon, double interest_rate, double maturity, int compound_times, Convention convention)
{
    auto dfs = get_discount_factors_0_T_less1(interest_rate, maturity, compound_times, convention);
    double fv = 0.0;
    for (const auto& df : dfs)
    {
        fv += (1.0 / df.df) * coupon;
    }
    return fv;
}

double fv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times, Convention convention, int g_compound_times, Convention g_convention)
{
    auto dfs = get_discount_factors_0_T_less1(interest_rate, maturity, compound_times, convention);
    double fv = 0.0;
    double i = 0.0;
    for (const auto& df : dfs)
    {
        fv += ((1.0 / df.df) * coupon) / zc2df(g, i, g_compound_times, g_convention);
        i += 1.0;
    }
    return fv;
}

double coupon_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times, Convention convention, int g_compound_times, Convention g_convention)
{
    double fv = fv_from_growth_coupon(coupon, g, interest_rate, maturity, compound_times, convention, g_compound_times, g_convention);
    return coupon_from_fv(fv, interest_rate, maturity, compound_times, convention);
}

double coupon_from_fv(double fv, double interest_rate, double maturity, int compound_times, Convention convention)
{
    auto dfs = get_discount_factors_0_T_less1(interest_rate, maturity, compound_times, convention);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += 1.0 / df.df;
    }
    return fv / total_df;
}

InterestRate on_capital(double initial, double final, double maturity, Convention convention, int compound_times)
{
    // cagr
    // return pow(final / initial, 1.0 / maturity) - 1.0;
    return InterestRate((final - initial) / initial, Convention::LINEAR).to_discount_factor(1.0).to_interest_rate(maturity, convention, compound_times);
}

TEST_CASE("bond_npv", "[fv]") {

    // Comparado con: https://mathcracker.com/es/calculadora-valor-bonos#results
    // valor presente de un bono
    // valorar un bono que da un yield "seguro" haciendo otros proyectos risk free
    double npv = bond_npv(16000,
        // inversion "segura" ofrecida por el bono
        100, 0.06,
        // inversion libre de riesgo en el mercado
        20, 1, Convention::YIELD);

    REQUIRE(npv == Catch::Approx(6135.87));
}

TEST_CASE("fv_from_coupon", "[fv]") {

    // Ahorro inicial en el futuro

    double initial = 10000;
    double r = 0.07;
    double maturity = 8;
    double fv1 = to_future_value(initial, InterestRate(r), maturity);
    double aportado1 = initial;
    double presente1 = to_present_value(initial, InterestRate(r), 0);

    REQUIRE(aportado1 == Catch::Approx(10000));
    REQUIRE(presente1 == Catch::Approx(10000));
    REQUIRE(fv1 == Catch::Approx(17181.8617983192));

    // Ahorro periodico (anual)

    double cuota = 5000;
    double fv2 = fv_from_coupon(5000, r, maturity, 1, Convention::YIELD);
    double aportado2 = real_from_coupon(5000, maturity, 1, Convention::YIELD);
    double presente2 = npv_from_coupon(5000, r, maturity, 1, Convention::YIELD);

    REQUIRE(aportado2 == Catch::Approx(40000.0));
    REQUIRE(presente2 == Catch::Approx(29856.4925310687));
    REQUIRE(fv2 == Catch::Approx(51299.0128451372));

    // Ahorro periodico (mensual)

    cuota = 1000;
    double compound_times = 12;
    double fv3 = fv_from_coupon(1000, r, maturity, compound_times, Convention::YIELD);
    double aportado3 = real_from_coupon(1000, maturity, compound_times, Convention::YIELD);
    double presente3 = npv_from_coupon(1000, r, maturity, compound_times, Convention::YIELD);

    REQUIRE(presente3 == Catch::Approx(6112.297390453));
    REQUIRE(aportado3 == Catch::Approx(96000.0));
    REQUIRE(fv3 == Catch::Approx(128198.8210340072));

    double final;
    double presente_total;
    initial = aportado1 + aportado2 + aportado3;
    presente_total = presente1 + presente2 + presente3;
    final = fv1 + fv2 + fv3;
    REQUIRE(presente_total == Catch::Approx(45968.7899215216));
    REQUIRE(coupon_from_real(initial, maturity, 12, Convention::YIELD) == Catch::Approx(1520.8333333333));
    REQUIRE(coupon_from_real(initial, maturity, 1, Convention::YIELD) == Catch::Approx(18250.0));
    REQUIRE(initial == Catch::Approx(146000.0));
    REQUIRE(final == Catch::Approx(196679.6956774635));

    double r_invest = on_capital(initial, final, maturity).r;
    REQUIRE(r_invest == Catch::Approx(0.0379485678));
}

TEST_CASE("cagr_interest", "[fv]") {

    double initial = 11000;
    double final = 21000;
    double past_years = 2;
    double forward_years = 2;
    // past info
    double r = on_capital(initial, final, past_years, Convention::YIELD).r;
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    r = on_capital(initial, final, past_years).r;
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    // forward prediction
    REQUIRE(to_future_value(final, InterestRate(r, Convention::YIELD, 12), forward_years) == Catch::Approx(44524.0670913586));

    // trading
    initial = 5000;
    r = 0.10;
    double r_anual = equivalent_rate(r, 12, 1).r;
    double years = 3.0;
    REQUIRE(to_future_value(initial, InterestRate(r, Convention::YIELD, 12), years) == \
        Catch::Approx(to_future_value(initial, InterestRate(r_anual), years)));
}

TEST_CASE("df & zc", "[fv]") {

    REQUIRE(zc2df(df2zc(0.95, 3, 1, Convention::LINEAR), 3, 1, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 1, Convention::LINEAR), 3, 1, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 4, Convention::LINEAR), 3, 4, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 4, Convention::LINEAR), 3, 4, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 1, Convention::YIELD), 3, 1, Convention::YIELD) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 1, Convention::YIELD), 3, 1, Convention::YIELD) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 4, Convention::YIELD), 3, 4, Convention::YIELD) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 4, Convention::YIELD), 3, 4, Convention::YIELD) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 1, Convention::EXPONENTIAL), 3, 1, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 1, Convention::EXPONENTIAL), 3, 1, Convention::EXPONENTIAL) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 4, Convention::EXPONENTIAL), 3, 4, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 4, Convention::EXPONENTIAL), 3, 4, Convention::EXPONENTIAL) == Catch::Approx(0.05));
}

TEST_CASE("bond_npv2", "[fv]") {

    double cash = 17181.8617983192;
    double r = 0.07;
    double maturity = 8;
    REQUIRE(to_present_value(cash, InterestRate(r), maturity) == Catch::Approx(10000));

    // future value
    double fv = 51299.0128451372;
    REQUIRE(coupon_from_fv(fv, r, maturity, 1, Convention::YIELD) == Catch::Approx(5000));
    REQUIRE(fv_from_coupon(5000, r, maturity, 1, Convention::YIELD) == Catch::Approx(fv));

    // traer flujos futuros a presente
    double npv = npv_from_coupon(5000, r, maturity, 1, Convention::YIELD);
    REQUIRE(npv == Catch::Approx(29856.4925310687));

    // Traerme a presente flujos futuros anuales
    REQUIRE(coupon_from_npv(npv, r, maturity, 1, Convention::YIELD) == Catch::Approx(5000));

    REQUIRE(classic_npv(
        // inversion
        6000,
        // cuota
        200, 
        // free risk rate
        InterestRate(0.16),
        // years
        3) == Catch::Approx(-5550.8220919267));
}

TEST_CASE("real coupon", "[fv]") {

    double coupon_netflix = 9.9;
    double maturity = 10;
    double real = real_from_coupon(coupon_netflix, maturity, 12, Convention::YIELD);

    REQUIRE(real == Catch::Approx(1188.0));

    // dividendo 0.08, precio dinero 0.03

    double npv = stock_npv(1000, 0.08, 0.03, maturity, 1, Convention::YIELD);
    REQUIRE(npv == Catch::Approx(170.6040567355));

    double fv = stock_fv(1000, 0.08, 0.03, maturity, 1, Convention::YIELD);
    REQUIRE(fv == Catch::Approx(573.1939655735));

    double real2 = stock_real(1000, 0.08, 0.03, maturity, 1, Convention::YIELD);
    REQUIRE(real2 == Catch::Approx(-843.9163793441));

    // dividendo 0.08, precio dinero 0.12

    double npv_ = stock_npv(1000, 0.08, 0.12, maturity, 1, Convention::YIELD);
    REQUIRE(npv_ == Catch::Approx(-904.0356845457));

    double fv_ = stock_fv(1000, 0.08, 0.12, maturity, 1, Convention::YIELD);
    REQUIRE(fv_ == Catch::Approx(-701.9494027814));

    double real2_ = stock_real(1000, 0.08, 0.12, maturity, 1, Convention::YIELD);
    REQUIRE(real2_ == Catch::Approx(-1705.8482083442));

    REQUIRE(on_capital(npv, fv, maturity, Convention::EXPONENTIAL).r == Catch::Approx(0.1211878754));
    REQUIRE(on_capital(npv_, fv_, maturity, Convention::EXPONENTIAL).r == Catch::Approx(-0.0253007508));
}

TEST_CASE("tn & te", "[fv]")
{
    double a = 0.05 / 12;
    // TASA NOMINAL a TASA EFECTIVA
    double b = equivalent_rate(0.05, 12, 1).r / 12;
    // TASA EFECTIVA A TASA NOMINAL
    double c = equivalent_rate(0.05, 1, 12).r / 12;

    double c1 = 1000 * a;
    double c2 = 1000 * b;
    double c3 = 1000 * c;

    REQUIRE(c1 == Catch::Approx(4.1666666667));
    REQUIRE(c2 == Catch::Approx(4.2634914901));
    REQUIRE(c3 == Catch::Approx(4.0741237836));

    // 5% reinvirtiendo 1 vez al añao
    REQUIRE(on_capital(1000, 1000 + (c1 * 12)).r == Catch::Approx(0.05));
    // 5% reinvirtiendo 12 veces equivalen a 5.1161% reinvirtiendo 1
    REQUIRE(on_capital(1000, 1000 + (c2 * 12)).r == Catch::Approx(0.0511618979));
    // 5% reinvirtiendo 1 vez equivalen a 4.888% reinvirtiendo 12
    REQUIRE(on_capital(1000, 1000 + (c3 * 12)).r == Catch::Approx(0.0488894854));

    //REQUIRE(tn_2_te(0.05, 12) == Catch::Approx(0.0511618979));
    REQUIRE(equivalent_rate(0.05, 12, 1).r == Catch::Approx(0.0511618979));

    //REQUIRE(te_2_tn(0.05, 12) == Catch::Approx(0.0488894854));
    REQUIRE(equivalent_rate(0.05, 1, 12).r == Catch::Approx(0.0488894854));

    REQUIRE(equivalent_rate(0.0488894854, 12, 1).r == Catch::Approx(0.05));
    REQUIRE(equivalent_rate(0.0511618979, 1, 12).r == Catch::Approx(0.05));

    REQUIRE(equivalent_rate(0.01, 365, 1).r == Catch::Approx(0.0100500287));
    REQUIRE(equivalent_rate(0.01, 1, 365).r == Catch::Approx(0.0099504665));

    /*
    10% mensual con reinversion mensual
    */
    double  fv = to_future_value(1000, InterestRate(0.10 * 12, Convention::YIELD, 12), 1);
    REQUIRE(fv == Catch::Approx(3138.428376721));
    REQUIRE(on_capital(1000, fv).r == Catch::Approx(equivalent_rate(0.10 * 12, 12, 1).r));

    /*
    10% mensual con reinversion anual = 120%
    */
    double  fv2 = to_future_value(1000, InterestRate(0.10 * 12), 1);
    REQUIRE(fv2 == Catch::Approx(2200.0));
    REQUIRE(on_capital(1000, fv2).r == Catch::Approx(equivalent_rate(0.10 * 12, 1, 1).r));

    /*
    2% semanal con reinversion semanal = 191.34%
    */
    double  fv3 = to_future_value(1000, InterestRate(0.02 * 54, Convention::YIELD, 54), 1);
    REQUIRE(fv3 == Catch::Approx(2913.4614441403));
    REQUIRE(on_capital(1000, fv3).r == Catch::Approx(InterestRate(0.02 * 54, Convention::YIELD, 54).to_other_interest_rate(Convention::YIELD).r));

    /*
    2% semanal con reinversion continua = 194.46%
    */
    double  fv4 = to_future_value(1000, InterestRate(0.02 * 54, Convention::EXPONENTIAL), 1);
    REQUIRE(fv4 == Catch::Approx(2944.6795510655));
    // ¿Como calcular ese CAGR?
    REQUIRE(on_capital(1000, fv4).r == Catch::Approx(InterestRate(0.02 * 54, Convention::EXPONENTIAL).to_other_interest_rate(Convention::YIELD).r));

    REQUIRE(equivalent_rate(0.05, 1, 12) == equivalent_rate(0.05, 1, Convention::YIELD, 12, Convention::YIELD));

    InterestRate other_r = InterestRate(0.2).to_other_interest_rate(Convention::EXPONENTIAL);
    REQUIRE(other_r.r == Catch::Approx(0.1823215568));
}

TEST_CASE("coupon growth", "[fv]")
{
    // el dividendo no crece
    double npv1 = npv_from_coupon(1000, 0.08, 5, 1, Convention::YIELD);
    REQUIRE(npv1 == Catch::Approx(3992.7100370781));

    // reinvertir anualmente
    double npv2 = npv_from_growth_coupon(1000, 0.05, 0.08, 5, 1, Convention::YIELD);
    REQUIRE(npv2 == Catch::Approx(4379.4737959505));

    // reinvertir mensualmente
    double npv3 = npv_from_growth_coupon(1000, 0.05, 0.08, 5, 12, Convention::YIELD, 12, Convention::YIELD);
    REQUIRE(npv3 == Catch::Approx(23219.4483321569));

    double fv1 = fv_from_growth_coupon(1000, 0.05, 0.08, 5, 12, Convention::YIELD, 12, Convention::YIELD);
    REQUIRE(fv1 == Catch::Approx(494045.000163533));

    double coupon1 = coupon_from_fv(fv1, 0.08, 1, 12, Convention::YIELD);
    REQUIRE(coupon1 == Catch::Approx(39682.5651273088));

    double fv2 = fv_from_coupon(3375.3053086363, 0.05, 5, 12, Convention::YIELD);
    REQUIRE(fv2 == Catch::Approx(229541.2924322575));

    double fixed_coupon = coupon_from_growth_coupon(1000, 0.05, 0.08, 5, 12, Convention::YIELD, 12, Convention::YIELD);
    REQUIRE(fixed_coupon == Catch::Approx(6723.8178851117));
}
