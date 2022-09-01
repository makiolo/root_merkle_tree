// https://www.linkedin.com/pulse/python-bootstrapping-zero-curve-sheikh-pancham#:~:text=The%20objective%20of%20bootstrapping%20is,the%20end%20of%20its%20term.

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <catch_amalgamated.hpp>
// header only date: https://raw.githubusercontent.com/HowardHinnant/date/master/include/date/date.h
// date algorithms: http://howardhinnant.github.io/date_algorithms.html
// date doc: https://howardhinnant.github.io/date.html
#include "date.h"

class DiscountFactor;
class Maturity;
class Calendar;
class InterestRate;
class CashFlow;
class StartCashFlow;
class EndCashFlow;

enum Convention
{
    LINEAR,
    YIELD,
    EXPONENTIAL,
};

enum Frequency
{
    ANNUAL = 1,
    SEMIANNUAL = 2,
    QUATERLY = 4,
    MONTHLY = 12,
};

enum DayCountConvention
{
    ACT_ACT,
    ACT_360,
    ACT_365,
    EQUALS,
};

// convert value and zc
double df2zc(double df, double maturity, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double zc2df(double zc, double maturity, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL);
InterestRate equivalent_rate(double rate, Convention conv, int compound_times, Convention other_convention = Convention::YIELD, int other_compound_times = Frequency::ANNUAL);
InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times = Frequency::ANNUAL);

// one cash flow
double to_present_value(double cash, const InterestRate& r, const Maturity& maturity);
double to_future_value(double cash, const InterestRate& r, const Maturity& maturity);
double to_present_value(double cash, const InterestRate& r, const Calendar& cal);
double to_future_value(double cash, const InterestRate& r, const Calendar& cal);

// fv - coupon - fv
double npv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal);
double coupon_from_npv(double npv, const InterestRate& interest_rate, const Calendar& cal);
double coupon_from_fv(double fv, const InterestRate& interest_rate, const Calendar& cal);
double fv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal);

// growth_coupon
double npv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal);
double fv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal);
double coupon_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal);

/*
growth_coupon_from_npv
growth_coupon_from_coupon
grouth_coupon_from_fv
*/
double compute_irr(const std::vector<double>& cf, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

// value products
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, const Calendar& cal);
double bond_npv(double face_value, double coupon, const InterestRate& interest_rate, const Calendar& cal);
double stock_npv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal);
double stock_fv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal);

// calcular value
InterestRate on_capital(double initial, double final_value, double maturity = 1.0, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);


struct Maturity
{
    Maturity(double value_)
        : value(value_)
    {
        ;
    }

    explicit Maturity(const date::year_month_day& pillar_, double value_)
        : pillar(pillar_)
        , value(value_)
    {
        ;
    }

    date::year_month_day pillar{};
    double value;
};

struct Period
{
    Period(const Maturity& start_, const Maturity& end_)
        : start(start_)
        , end(end_)
    {

    }

    Maturity start;
    Maturity end;
};

struct Calendar
{
    date::year_month_day start_date;
    date::year_month_day end_date;
    int tenor;
    DayCountConvention dc_convention;

    Calendar(const date::year_month_day& start_date_, const date::year_month_day& end_date_, int tenor_ = 12, DayCountConvention dc_convention_ = DayCountConvention::EQUALS)
        : start_date(start_date_)
        , end_date(end_date_)
        , tenor(tenor_)
        , dc_convention(dc_convention_)
    {

    }

    Calendar(const date::year_month_day& start_date_, int duration, int tenor_ = 12, DayCountConvention dc_convention_ = DayCountConvention::EQUALS)
        : start_date(start_date_)
        , end_date(start_date_ + date::years(duration))
        , tenor(tenor_)
        , dc_convention(dc_convention_)
    {

    }

    [[nodiscard]] std::vector<Period> generate(bool skip_first = false) const
    {
        using namespace date;

        std::vector<Period> data;
        auto pillar_day = start_date;
        int i = 0;
        if (!skip_first)
        {
            pillar_day += months(tenor);
            i++;
        }
        while ((skip_first && (pillar_day < end_date)) || (!skip_first && (pillar_day <= end_date)))
        {
            double count;
            switch (dc_convention)
            {
            case DayCountConvention::ACT_ACT:
            {
                double m = double((sys_days{ jan / day(1) / (pillar_day.year() + years(1)) } - sys_days{ jan / day(1) / pillar_day.year() }).count());
                count = double((sys_days{ pillar_day } - sys_days{ start_date }).count()) / m;
                break;
            }
            case DayCountConvention::ACT_365:
            {
                count = double((sys_days{ pillar_day } - sys_days{ start_date }).count()) / 365.0;
                break;
            }
            case DayCountConvention::EQUALS:
            {
                count = double(i) / (12.0 / tenor);
                break;
            }
            case DayCountConvention::ACT_360:
            default:
            {
                count = double((sys_days{ pillar_day } - sys_days{ start_date }).count()) / 360.0;
                break;
            }
            }
            auto start = Maturity{ pillar_day, count };
            pillar_day += months(tenor);
            auto end = Maturity{ pillar_day, count };
            data.emplace_back(start, end);
            i += 1;
        }
        return data;
    }

    [[nodiscard]] Period get_last_period() const
    {
        auto mats = generate(false);
        return mats.back();
    }
};

class DiscountFactor
{
public:
    explicit DiscountFactor(double value_)
        : value(value_)
    {
        ;
    }

    DiscountFactor(const DiscountFactor& other)
    {
        value = other.value;
    }

    DiscountFactor(DiscountFactor&& other) noexcept
    {
        value = other.value;
    }

    [[nodiscard]] InterestRate to_interest_rate(double maturity, Convention convention_ = Convention::YIELD, int compound_times_ = Frequency::ANNUAL) const;

public:
    double value;
};

class InterestRate
{
public:
    explicit InterestRate(double value_, Convention convention_ = Convention::YIELD, int compound_times_ = 1)
        : value(value_)
        , conv(convention_)
        , c(compound_times_)
    {
        ;
    }

    InterestRate(const InterestRate& other)
    {
        value = other.value;
        c = other.c;
        conv = other.conv;
    }

    InterestRate(InterestRate&& other) noexcept
    {
        value = other.value;
        c = other.c;
        conv = other.conv;
    }

    bool operator==(const InterestRate& rhs) const
    {
        return value == rhs.value &&
            c == rhs.c &&
            conv == rhs.conv;
    }

    [[nodiscard]] std::vector<DiscountFactor> get_discount_factors(const Calendar& cal) const
    {
        std::vector<DiscountFactor> dfs;
        for (auto& period : cal.generate(false))
        {
            dfs.push_back(to_discount_factor(period.end));
        }
        return dfs;
    }

    [[nodiscard]] std::vector<DiscountFactor> get_discount_factors_skip_first(const Calendar& cal) const
    {
        std::vector<DiscountFactor> dfs;
        for (auto& period : cal.generate(true))
        {
            dfs.push_back(to_discount_factor(period.start));
        }
        return dfs;
    }

    [[nodiscard]] Maturity get_first_maturity(const Calendar& cal) const
    {
        auto mats = cal.generate(true);
        return mats.front().start;
    }

    [[nodiscard]] DiscountFactor to_discount_factor(const Maturity& maturity) const
    {
        return DiscountFactor(zc2df(value, maturity.value, conv, c));
    }

    [[nodiscard]] InterestRate to_other_interest_rate(Convention other_convention, int other_compound_times = Frequency::ANNUAL) const
    {
        return equivalent_rate(value, conv, c, other_convention, other_compound_times);
    }

    [[nodiscard]] DiscountFactor direct_discount(const Maturity& one, const Maturity& other) const;
    [[nodiscard]] InterestRate forward_rate(const Maturity& one, const Maturity& other) const;
    [[nodiscard]] DiscountFactor next_discount(const Maturity& one, const InterestRate& forward_rate, double m = 1.0) const;

public:
    double value;  // annual rate
    int c;  // reinversions each year
    Convention conv;  // convention

    static const InterestRate ZERO;
};

const InterestRate InterestRate::ZERO(0.0);


DiscountFactor InterestRate::direct_discount(const Maturity& one, const Maturity& other) const
{
    double df0 = to_discount_factor(one).value;
    double df1 = to_discount_factor(other).value;

    return DiscountFactor(df1 / df0);
}

InterestRate InterestRate::forward_rate(const Maturity& one, const Maturity& other) const
{
    double df0 = to_discount_factor(one).value;
    double df1 = to_discount_factor(other).value;

    double m = other.value - one.value;

    return InterestRate((df0 / df1 - 1.0) / m);
}

DiscountFactor InterestRate::next_discount(const Maturity& one, const InterestRate& forward_rate, double m) const
{
    double discount = to_discount_factor(one).value;

    return DiscountFactor(discount / (1.0 + m * forward_rate.value));
}

struct Engine
{
    // TODO:
};

struct Model
{
    // TODO:
};

struct Legs
{
    std::vector<CashFlow> flows;
};

struct Product
{
    Calendar cal;
    std::vector<Legs> legs;
};

class CashFlow
{
public:
    explicit CashFlow(const Calendar& cal_, const InterestRate& ir_, double cash_)
        : cal(cal_)
        , ir(ir_)
        , cash(cash_)
    {
        ;
    }

public:
    Calendar cal;
    InterestRate ir;
    double cash;
};

class CouponCashFlow : public CashFlow
{
public:
    CouponCashFlow(const Calendar& cal_, const InterestRate& ir_, double cash_, const InterestRate& growth_ = InterestRate::ZERO)
        : CashFlow(cal_, ir_, cash_)
        , growth(growth_)
    {

    }

    [[nodiscard]] StartCashFlow to_start_cashflow() const;
    [[nodiscard]] EndCashFlow to_end_cashflow() const;
    // double to_custom_cashflow() const
public:
    InterestRate growth;
};

class StartCashFlow : public CashFlow
{
public:
    StartCashFlow(const Calendar& cal_, const InterestRate& ir_, double cash_)
        : CashFlow(cal_, ir_, cash_)
    {
        ;
    }

    [[nodiscard]] CouponCashFlow to_coupon() const
    {
        auto cash2 = coupon_from_npv(cash, ir, cal);
        return CouponCashFlow{ cal, ir, cash2 };
    }

    [[nodiscard]] EndCashFlow to_end_cashflow() const;

    /*
    double to_custom_ca1shflow() const
    {
        // TODO
    }
    */
};

class EndCashFlow : public CashFlow
{
public:
    EndCashFlow(const Calendar& cal_, const InterestRate& ir_, double cash_)
        : CashFlow(cal_, ir_, cash_)
    {
        ;
    }

    [[nodiscard]] CouponCashFlow to_coupon() const
    {
        auto cash2 = coupon_from_fv(cash, ir, cal);
        return CouponCashFlow{ cal, ir, cash2 };
    }

    [[nodiscard]] StartCashFlow to_start_cashflow() const
    {
        auto cash2 = to_present_value(cash, ir, cal);
        return StartCashFlow{ cal, ir, cash2 };
    }

    /*
    double to_custom_cashflow() const
    {
        return coupon_from_fv(cash, ir, cal);
    }
    */
};

class CustomCashFlow : public CashFlow
{
public:
    CustomCashFlow(const Calendar& cal_, const InterestRate& ir_, double cash_, const Maturity& maturity_)
        : CashFlow(cal_, ir_, cash_)
        , maturity(maturity_)
    {
        ;
    }

    // to_coupon
    [[nodiscard]] StartCashFlow to_start_cashflow() const
    {
        auto cash2 = to_present_value(cash, ir, maturity);
        return StartCashFlow{ cal, ir, cash2 };
    }
    [[nodiscard]] EndCashFlow to_end_cashflow() const
    {
        auto cash2 = to_future_value(cash, ir, maturity);
        return EndCashFlow{ cal, ir, cash2 };
    }

protected:
    Maturity maturity;
};

InterestRate DiscountFactor::to_interest_rate(double maturity, Convention convention_, int compound_times_) const
{
    return InterestRate(df2zc(value, maturity, convention_, compound_times_), convention_, compound_times_);
}

[[nodiscard]] StartCashFlow CouponCashFlow::to_start_cashflow() const
{
    auto cash2 = npv_from_growth_coupon(cash, growth, ir, cal);
    return StartCashFlow{ cal, ir, cash2 };
}

[[nodiscard]] EndCashFlow CouponCashFlow::to_end_cashflow() const
{
    auto cash2 = fv_from_growth_coupon(cash, growth, ir, cal);
    return EndCashFlow{ cal, ir, cash2 };
}

[[nodiscard]] EndCashFlow StartCashFlow::to_end_cashflow() const
{
    auto cash2 = to_future_value(cash, ir, cal);
    return EndCashFlow{ cal, ir, cash2 };
}
// ********************** //

double df2zc(double df, double maturity, Convention conv, int compound_times)
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

double zc2df(double zc, double maturity, Convention conv, int compound_times)
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

#define LOW_RATE  (-0.999)
#define HIGH_RATE  0.999
#define MAX_ITERATION 1000
#define PRECISION_REQ 0.00000001
double compute_irr(const std::vector<double>& cf, Convention convention, int compound_times)
{
    int i = 0, j = 0;
    double m = 0.0;
    double old = 0.00;
    double new_ = 0.00;
    double oldguessRate = LOW_RATE;
    double newguessRate = LOW_RATE;
    double guessRate = LOW_RATE;
    double lowGuessRate = LOW_RATE;
    double highGuessRate = HIGH_RATE;
    double npv = 0.0;
    double discount_factor = 0.0;
    for (i = 0; i < MAX_ITERATION; i++)
    {
        npv = 0.00;
        for (j = 0; j < cf.size(); j++)
        {
            discount_factor = zc2df(guessRate, j, convention, compound_times);
            npv = npv + (cf[j] * discount_factor);
        }
        /* Stop checking once the required precision is achieved */
        if ((npv > 0) && (npv < PRECISION_REQ))
            break;
        if (old == 0)
            old = npv;
        else
            old = new_;
        new_ = npv;
        if (i > 0)
        {
            if (old < new_)
            {
                if (old < 0 && new_ < 0)
                    highGuessRate = newguessRate;
                else
                    lowGuessRate = newguessRate;
            }
            else
            {
                if (old > 0 && new_ > 0)
                    lowGuessRate = newguessRate;
                else
                    highGuessRate = newguessRate;
            }
        }
        oldguessRate = guessRate;
        guessRate = (lowGuessRate + highGuessRate) / 2;
        newguessRate = guessRate;
    }
    return guessRate;
}


InterestRate equivalent_rate(double rate, Convention convention, int compound_times, Convention other_convention, int other_compound_times)
{
    return InterestRate(rate, convention, compound_times)
        .to_discount_factor(Maturity(1.0))
        .to_interest_rate(1.0, other_convention, other_compound_times);
}

InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times)
{
    return equivalent_rate(rate, Convention::YIELD, compound_times, Convention::YIELD, other_compound_times);
}

// tenemos un cash en "maturity" y nos lo traemos a "0"
double to_present_value(double cash, const InterestRate& r, const Maturity& maturity)
{
    return cash * r.to_discount_factor(maturity).value;
}

// tenemos un cash en "0" y nos lo traemos a "maturity"
double to_future_value(double cash, const InterestRate& r, const Maturity& maturity)
{
    return cash / r.to_discount_factor(maturity).value;
}

// tenemos un cash al final del calendario y nos lo traemos a "0"
double to_present_value(double cash, const InterestRate& r, const Calendar& cal)
{
    auto maturity = cal.get_last_period().end; // obtener maturity del cash (teniendo cierto "cal")
    return to_present_value(cash, r, maturity);
}

// tenemos un en 0 y nos lo traemos al final del calendario
double to_future_value(double cash, const InterestRate& r, const Calendar& cal)
{
    auto maturity = cal.get_last_period().end; // obtener maturity del cash (teniendo cierto "cal")
    return to_future_value(cash, r, maturity);
}

// only coupons
double npv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    return npv_from_growth_coupon(coupon, InterestRate::ZERO, interest_rate, cal);
}

double npv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal)
{
    auto dfs = interest_rate.get_discount_factors(cal);

    double npv = 0.0;
    double i = 0.0;
    for (const auto& df : dfs)
    {
        // auto denominator = zc2df(growth_rate.value, i, growth_rate.conv, growth_rate.c);

        npv += (df.value * (coupon / interest_rate.c)) / growth_rate.to_discount_factor(i).value;
        i += 1.0;
    }
    return npv;
}

double fv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal)
{
    double npv = npv_from_growth_coupon(coupon, growth_rate, interest_rate, cal);
    return to_future_value(npv, interest_rate, cal);
}

double fv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    return fv_from_growth_coupon(coupon, InterestRate(0.0), interest_rate, cal);
}

double coupon_from_npv(double npv, const InterestRate& interest_rate, const Calendar& cal)
{
    using namespace date;

    auto dfs = interest_rate.get_discount_factors(cal);

    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += df.value;
    }
    return (npv * interest_rate.c) / total_df;
}

// VAN = coupons - initial investement
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    return npv_from_coupon(coupon, interest_rate, cal) - investment;
}

// coupons + payment on yield-maturity
double bond_npv(double face_value, double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    double coupons = npv_from_coupon(coupon, interest_rate, cal);
    double yield_on_payment = to_present_value(face_value, interest_rate, cal);
    return coupons + yield_on_payment;
}

// stock investment = coupons + payment on yield-maturity - initial investement
double stock_npv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal)
{
    double coupon = investment * (dividend - interest_rate.value);
    double coupons = npv_from_coupon(coupon, interest_rate, cal);
    double yield_on_payment = to_present_value(investment, interest_rate, cal);
    double npv = coupons + yield_on_payment - investment;
    return npv;
}

// TODO: reimplement it
double stock_fv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal)
{
    double coupon = investment * (dividend - interest_rate.value);
    double coupons = fv_from_coupon(coupon, interest_rate, cal);
    double npv = coupons;
    return npv;
}

double coupon_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal)
{
    double npv = npv_from_growth_coupon(coupon, growth_rate, interest_rate, cal);
    return coupon_from_npv(npv, interest_rate, cal);
}

double coupon_from_fv(double fv, const InterestRate& interest_rate, const Calendar& cal)
{
    auto dfs = interest_rate.get_discount_factors_skip_first(cal);

    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += 1.0 / df.value;
    }
    return (fv * interest_rate.c) / total_df;
}

InterestRate on_capital(double initial, double final_value, double maturity, Convention convention, int compound_times)
{
    if (convention == Convention::YIELD && compound_times == Frequency::ANNUAL)
    {
        // cagr
        return InterestRate(pow(final_value / initial, 1.0 / maturity) - 1.0);
    }
    else
    {
        return InterestRate((final_value - initial) / initial, Convention::LINEAR)
            .to_discount_factor(Maturity(1.0))
            .to_interest_rate(maturity, convention, compound_times);
    }
}

TEST_CASE("bond_npv", "[fv]") {

    using namespace date;

    // Comparado con: https://mathcracker.com/es/calculadora-valor-bonos#results
    // valor presente de un bono
    // valorar un bono que da un yield "seguro" haciendo otros proyectos risk free
    double npv = bond_npv(
        // face value
        16000,
        // cupon
        100,
        InterestRate(0.06),
        // calendar
        Calendar(2022_y / 1 / 1, 20));

    REQUIRE(npv == Catch::Approx(6135.87));
}

TEST_CASE("fv_from_coupon", "[fv]") {

    using namespace date;

    // Ahorro inicial en el futuro

    double initial = 10000;
    double r = 0.07;
    int maturity = 8;
    auto cal = Calendar(2022_y / 1 / 1, maturity);
    double fv1 = to_future_value(initial, InterestRate(r), cal);
    double aportado1 = initial;
    double presente1 = initial;

    REQUIRE(aportado1 == Catch::Approx(10000));
    REQUIRE(presente1 == Catch::Approx(10000));
    REQUIRE(fv1 == Catch::Approx(17181.8617983192));

    // Ahorro periodico (anual)
    double cuota;

    cuota = 5000;
    //double aportado2 = real_from_coupon(cuota, maturity, Convention::YIELD);
    double presente2 = npv_from_coupon(cuota, InterestRate(r), cal);
    double fv2 = fv_from_coupon(cuota, InterestRate(r), cal);

    // REQUIRE(aportado2 == Catch::Approx(40000.0));
    REQUIRE(presente2 == Catch::Approx(29856.4925310687));
    REQUIRE(fv2 == Catch::Approx(51299.0128451372));

    // Ahorro periodico (mensual)

    cuota = 1000;
    int compound_times = 12;
    auto cal2 = Calendar(2022_y / 1 / 1, maturity, 1);
    //double aportado3 = real_from_coupon(cuota, maturity, Convention::YIELD, compound_times);
    double presente3 = npv_from_coupon(cuota * compound_times, InterestRate(r, Convention::YIELD, compound_times), cal2);
    double fv3 = fv_from_coupon(cuota * compound_times, InterestRate(r, Convention::YIELD, compound_times), cal2);

    REQUIRE(presente3 == Catch::Approx(73347.5686854354));
    // REQUIRE(aportado3 == Catch::Approx(96000.0));
    REQUIRE(fv3 == Catch::Approx(128198.8210340072));

    double final_value;
    double presente_total;
    double aportado_total = aportado1; // + aportado2 + aportado3;
    presente_total = presente1 + presente2 + presente3;
    final_value = fv1 + fv2 + fv3;
    //REQUIRE(coupon_from_real(aportado_total, maturity, Convention::YIELD, 12) == Catch::Approx(1520.8333333333));
    //REQUIRE(coupon_from_real(aportado_total, maturity, Convention::YIELD) == Catch::Approx(18250.0));
    REQUIRE(presente_total == Catch::Approx(113204.0612165041));
    REQUIRE(aportado_total == Catch::Approx(10000));
    REQUIRE(final_value == Catch::Approx(196679.6956774635));

    InterestRate r_invest = on_capital(aportado_total, final_value, maturity);
    REQUIRE(r_invest.value == Catch::Approx(0.4511755111));
}

TEST_CASE("fv_from_coupon2", "[fv]")
{
    // Ahorro periodico (semanal)
    using namespace date;

    double cuota = 200;
    double frecuencia = 54;
    int maturity = 3.0;
    double r = 0.08;
    auto cal = Calendar(2022_y / 1 / 1, maturity);
    double presente = npv_from_coupon(cuota * frecuencia, InterestRate(r), cal);
    //double aportado = real_from_coupon(cuota * frecuencia, maturity, Convention::YIELD);
    double future = fv_from_coupon(cuota * frecuencia, InterestRate(r), cal);

    REQUIRE(presente == Catch::Approx(27832.6474622771));
    //REQUIRE(aportado == Catch::Approx(32400.0));
    REQUIRE(future == Catch::Approx(35061.12));
}

TEST_CASE("cagr_interest", "[fv]") {

    double initial = 11000;
    double final = 21000;
    double past_years = 2;
    double forward_years = 2;
    // past info
    double r = on_capital(initial, final, past_years, Convention::YIELD).value;
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    r = on_capital(initial, final, past_years).value;
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    // forward prediction
    REQUIRE(to_future_value(final, InterestRate(r, Convention::YIELD, 12), forward_years) == Catch::Approx(44524.0670913586));

    // trading
    initial = 5000;
    r = 0.10;
    double r_anual = equivalent_rate(r, 12, 1).value;
    double years = 3.0;
    REQUIRE(to_future_value(initial, InterestRate(r, Convention::YIELD, 12), years) == \
        Catch::Approx(to_future_value(initial, InterestRate(r_anual), years)));
}

TEST_CASE("value & zc", "[fv]") {

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::LINEAR, 4), 3, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::YIELD), 3, Convention::YIELD) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::YIELD), 3, Convention::YIELD) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::YIELD, 4), 3, Convention::YIELD, 4) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::YIELD, 4), 3, Convention::YIELD, 4) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.05));
}

TEST_CASE("bond_npv2", "[fv]") {

    using namespace date;

    double cash = 17181.8617983192;
    double r = 0.07;
    int maturity = 8;
    auto cal = Calendar(2022_y / 1 / 1, maturity);
    REQUIRE(to_present_value(cash, InterestRate(r), cal) == Catch::Approx(10000));

    // future value
    double fv = 51299.0128451372;
    REQUIRE(coupon_from_fv(fv, InterestRate(r), cal) == Catch::Approx(5000));
    REQUIRE(fv_from_coupon(5000, InterestRate(r), cal) == Catch::Approx(fv));

    // traer flujos futuros a presente
    double npv = npv_from_coupon(5000, InterestRate(r), cal);
    REQUIRE(npv == Catch::Approx(29856.4925310687));

    // Traerme a presente flujos futuros anuales
    REQUIRE(coupon_from_npv(npv, InterestRate(r), cal) == Catch::Approx(5000));

    REQUIRE(classic_npv(
        // inversion
        6000,
        // cuota
        500,
        // free risk rate
        InterestRate(0.01),
        // years
        Calendar(2022_y / 1 / 1, 1)) == Catch::Approx(-5504.9504950495));

    double npv1 = classic_npv(1000, 100, InterestRate(-0.1940185202), Calendar(2022_y / 1 / 1, 6));
    REQUIRE(npv1 == Catch::Approx(364.7956282082));

    std::vector<double> cf = { -1000, 100, 100, 100, 100, 100 };
    double irr = compute_irr(cf);
    REQUIRE(irr == Catch::Approx(-0.1940185202));
}

TEST_CASE("real coupon", "[fv]") {

    using namespace date;

    double coupon_netflix = 9.9;
    int maturity = 10;

    auto cal = Calendar(2022_y / 1 / 1, maturity);

    //double real = real_from_coupon(coupon_netflix, maturity, Convention::YIELD, 12);
    //REQUIRE(real == Catch::Approx(1188.0));

    // dividendo 0.08, precio dinero 0.03

    double npv = stock_npv(1000, 0.08, InterestRate(0.03), cal);
    REQUIRE(npv == Catch::Approx(170.6040567355));

    double fv = stock_fv(1000, 0.08, InterestRate(0.03), cal);
    REQUIRE(fv == Catch::Approx(573.1939655735));

    // double real2 = stock_real(1000, 0.08, 0.03, maturity, Convention::YIELD);
    // REQUIRE(real2 == Catch::Approx(-843.9163793441));

    // dividendo 0.08, precio dinero 0.12

    double npv_ = stock_npv(1000, 0.08, InterestRate(0.12), cal);
    REQUIRE(npv_ == Catch::Approx(-904.0356845457));

    double fv_ = stock_fv(1000, 0.08, InterestRate(0.12), cal);
    REQUIRE(fv_ == Catch::Approx(-701.9494027814));

    // double real2_ = stock_real(1000, 0.08, 0.12, maturity, Convention::YIELD);
    // REQUIRE(real2_ == Catch::Approx(-1705.8482083442));

    REQUIRE(on_capital(npv, fv, maturity, Convention::EXPONENTIAL).value == Catch::Approx(0.1211878754));
    REQUIRE(on_capital(npv_, fv_, maturity, Convention::EXPONENTIAL).value == Catch::Approx(-0.0253007508));
}

TEST_CASE("tn & te", "[fv]")
{
    double a = 0.05 / 12;
    // TASA NOMINAL a TASA EFECTIVA
    double b = equivalent_rate(0.05, 12, 1).value / 12;
    // TASA EFECTIVA A TASA NOMINAL
    double c = equivalent_rate(0.05, 1, 12).value / 12;

    double c1 = 1000 * a;
    double c2 = 1000 * b;
    double c3 = 1000 * c;

    REQUIRE(c1 == Catch::Approx(4.1666666667));
    REQUIRE(c2 == Catch::Approx(4.2634914901));
    REQUIRE(c3 == Catch::Approx(4.0741237836));

    // 5% reinvirtiendo 1 vez al añao
    REQUIRE(on_capital(1000, 1000 + (c1 * 12)).value == Catch::Approx(0.05));
    // 5% reinvirtiendo 12 veces equivalen a 5.1161% reinvirtiendo 1
    REQUIRE(on_capital(1000, 1000 + (c2 * 12)).value == Catch::Approx(0.0511618979));
    // 5% reinvirtiendo 1 vez equivalen a 4.888% reinvirtiendo 12
    REQUIRE(on_capital(1000, 1000 + (c3 * 12)).value == Catch::Approx(0.0488894854));

    //REQUIRE(tn_2_te(0.05, 12) == Catch::Approx(0.0511618979));
    REQUIRE(equivalent_rate(0.05, 12, 1).value == Catch::Approx(0.0511618979));

    //REQUIRE(te_2_tn(0.05, 12) == Catch::Approx(0.0488894854));
    REQUIRE(equivalent_rate(0.05, 1, 12).value == Catch::Approx(0.0488894854));

    REQUIRE(equivalent_rate(0.0488894854, 12, 1).value == Catch::Approx(0.05));
    REQUIRE(equivalent_rate(0.0511618979, 1, 12).value == Catch::Approx(0.05));

    REQUIRE(equivalent_rate(0.01, 365, 1).value == Catch::Approx(0.0100500287));
    REQUIRE(equivalent_rate(0.01, 1, 365).value == Catch::Approx(0.0099504665));

    /*
    10% mensual con reinversion mensual
    */
    double  fv = to_future_value(1000, InterestRate(0.10 * 12, Convention::YIELD, 12), 1);
    REQUIRE(fv == Catch::Approx(3138.428376721));
    REQUIRE(on_capital(1000, fv).value == Catch::Approx(equivalent_rate(0.10 * 12, 12, 1).value));

    /*
    10% mensual con reinversion anual = 120%
    */
    double  fv2 = to_future_value(1000, InterestRate(0.10 * 12), 1);
    REQUIRE(fv2 == Catch::Approx(2200.0));
    REQUIRE(on_capital(1000, fv2).value == Catch::Approx(equivalent_rate(0.10 * 12, 1, 1).value));

    /*
    2% semanal con reinversion semanal = 191.34%
    */
    double  fv3 = to_future_value(1000, InterestRate(0.02 * 54, Convention::YIELD, 54), 1);
    REQUIRE(fv3 == Catch::Approx(2913.4614441403));
    REQUIRE(on_capital(1000, fv3).value == Catch::Approx(InterestRate(0.02 * 54, Convention::YIELD, 54).to_other_interest_rate(Convention::YIELD).value));

    /*
    2% semanal con reinversion continua = 194.46%
    */
    double  fv4 = to_future_value(1000, InterestRate(0.02 * 54, Convention::EXPONENTIAL), 1);
    REQUIRE(fv4 == Catch::Approx(2944.6795510655));
    // ¿Como calcular ese CAGR?
    REQUIRE(on_capital(1000, fv4).value == Catch::Approx(InterestRate(0.02 * 54, Convention::EXPONENTIAL).to_other_interest_rate(Convention::YIELD).value));

    REQUIRE(equivalent_rate(0.05, 1, 12) == equivalent_rate(0.05, Convention::YIELD, 1, Convention::YIELD, 12));

    InterestRate other_r = InterestRate(0.2).to_other_interest_rate(Convention::EXPONENTIAL);
    REQUIRE(other_r.value == Catch::Approx(0.1823215568));
}

TEST_CASE("coupon growth", "[fv]")
{
    using namespace date;

    auto cal = Calendar(2022_y / 1 / 1, 5);

    // el dividendo no crece
    double npv1 = npv_from_coupon(1000, InterestRate(0.08), cal);
    REQUIRE(npv1 == Catch::Approx(3992.7100370781));

    // reinvertir anualmente
    double npv2 = npv_from_growth_coupon(1000, InterestRate(0.05), InterestRate(0.08), cal);
    REQUIRE(npv2 == Catch::Approx(4379.4737959505));
}

TEST_CASE("coupon growth2", "[fv]")
{
    using namespace date;

    auto cal = Calendar(2022_y / 1 / 1, 5, 1);

    // npv y fv from growth cupon

    double npv_from_gcoupon = npv_from_growth_coupon(1000, InterestRate(0.05, Convention::YIELD, 12), InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(npv_from_gcoupon == Catch::Approx(23219.4483321569));

    double fv_from_gcoupon = fv_from_growth_coupon(1000, InterestRate(0.05, Convention::YIELD, 12), InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(fv_from_gcoupon == Catch::Approx(34593.3954467948));


    // cupon from growth cupon

    double fixed_coupon = coupon_from_growth_coupon(1000,
        InterestRate(0.05, Convention::YIELD, 12),
        InterestRate(0.08, Convention::YIELD, 12),
        cal);
    REQUIRE(fixed_coupon == Catch::Approx(5649.6802745071));

    // fv

    double coupon1 = coupon_from_fv(fv_from_gcoupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(coupon1 == Catch::Approx(fixed_coupon));

    double fv4 = fv_from_coupon(coupon1, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(fv4 == Catch::Approx(fv_from_gcoupon));

    double fv5 = fv_from_coupon(fixed_coupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(fv5 == Catch::Approx(fv_from_gcoupon));

    // npv

    double coupon2 = coupon_from_npv(npv_from_gcoupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(coupon2 == Catch::Approx(fixed_coupon));

    double npv4 = npv_from_coupon(coupon2, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(npv4 == Catch::Approx(npv_from_gcoupon));

    double npv5 = npv_from_coupon(fixed_coupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(npv5 == Catch::Approx(npv_from_gcoupon));
}

TEST_CASE("date C++20", "[date]")
{
    using namespace date;
    auto x = 2012_y / 1 / 24;
    auto y = 2013_y / 1 / 8;
    auto diff = (sys_days{ y } - sys_days{ x }).count();
    REQUIRE(diff == Catch::Approx(350));

    auto start_date = day(1) / jan / 2020;
    auto end_date = last / jan / 2030;
    double last_maturity;
    for (auto d = start_date; d < end_date; d += months(1))
    {
        // ACT/ACT
        int actual = (sys_days{ jan / day(1) / (d.year() + years(1)) } - sys_days{ jan / day(1) / d.year() }).count();
        double maturity = double((sys_days{ d } - sys_days{ start_date }).count()) / double(actual);
        std::cout << maturity << std::endl;
        std::cout << "for: " << d << " (dia " << d.day() << ")" << ": " << to_present_value(1000, InterestRate(0.05), maturity) << std::endl;
        last_maturity = maturity;
    }
    REQUIRE(last_maturity == Catch::Approx(9.9232876712));

    Calendar cal{ start_date, end_date, 3, DayCountConvention::EQUALS };

    std::vector<double> v1, v2;
    for (auto& period : cal.generate(true))
    {
        std::cout << "begin mode = true, pillar: " << period.start.pillar << " - value: " << period.start.value << std::endl;
        v1.push_back(period.start.value);
    }
    for (auto& period : cal.generate(false))
    {
        std::cout << "begin mode = false, pillar: " << period.end.pillar << " - value: " << period.end.value << std::endl;
        v2.push_back(period.end.value);
    }
    std::vector<double> m1, m2;
    int c = 4;
    for (int i = 0; i < 10 * c; ++i)
    {
        std::cout << "value: " << double(i) / c << std::endl;
        m1.push_back(double(i) / c);
    }
    for (int i = 1; i <= 10 * c; ++i)
    {
        std::cout << "value: " << double(i) / c << std::endl;
        m2.push_back(double(i) / c);
    }
    REQUIRE(v1 == m1);
    REQUIRE(v2 == m2);
}

TEST_CASE("forwards1", "[fw]")
{
    using namespace date;

    auto start_date = jan / day(1) / 2020;
    auto end_date = jan / day(1) / 2030;

    Calendar cal{ start_date, end_date, 12, DayCountConvention::EQUALS };
    auto fixings = cal.generate(true);

    InterestRate r(0.08);

    double cash = 1;

    double fwd2 = r.direct_discount(fixings[0].start, fixings[1].start).value;
    double fwd3 = r.direct_discount(fixings[1].start, fixings[2].start).value;
    double fwd4 = r.direct_discount(fixings[0].start, fixings[2].start).value;

    REQUIRE((fwd2) == Catch::Approx(0.9259259259));
    REQUIRE((fwd3) == Catch::Approx(0.9259259259));
    REQUIRE((fwd4) == Catch::Approx(0.9259259259 * 0.9259259259));

    double fwr1 = r.forward_rate(fixings[0].start, fixings[1].start).to_other_interest_rate(Convention::YIELD, 12).value / 12.0;
    double fwr2 = r.forward_rate(fixings[1].start, fixings[2].start).value;
    double fwr3 = r.forward_rate(fixings[2].start, fixings[3].start).value;

    REQUIRE((fwr1) == Catch::Approx(0.0064340301));
    REQUIRE((fwr2) == Catch::Approx(0.08));
    REQUIRE((fwr3) == Catch::Approx(0.08));

    REQUIRE(to_future_value(cash, r, fixings[0].start) == Catch::Approx(1.0));
    REQUIRE(to_future_value(cash, r, fixings[1].start) == Catch::Approx(1.08));
    REQUIRE(to_future_value(cash, r, fixings[2].start) == Catch::Approx(1.1664));
    REQUIRE(to_future_value(cash, r, cal) == Catch::Approx(2.1589249972728));

    REQUIRE((1.08 * fwd2) == Catch::Approx(1.0));
    REQUIRE((1.1664 * fwd3) == Catch::Approx(1.08));
    REQUIRE((1.259712 * fwd2 * fwd3) == Catch::Approx(1.08));

    REQUIRE(to_present_value(1.0, r, fixings[0].start) == Catch::Approx(1.0));
    REQUIRE(to_present_value(1.08, r, fixings[1].start) == Catch::Approx(1.0));
    REQUIRE(to_present_value(1.1664, r, fixings[2].start) == Catch::Approx(1.0));
    REQUIRE(to_present_value(2.1589249972728, r, cal) == Catch::Approx(1.0));

    double df0 = r.to_discount_factor(fixings[0].start).value;
    double df1 = r.to_discount_factor(fixings[1].start).value;
    double df2 = r.to_discount_factor(fixings[2].start).value;

    REQUIRE(df0 == Catch::Approx(1.0));
    REQUIRE(df1 == Catch::Approx(0.9259259259));
    REQUIRE(df2 == Catch::Approx(0.8573388203));

    REQUIRE(r.next_discount(fixings[1].start, InterestRate(fwr2), 1.0).value == Catch::Approx(r.direct_discount(fixings[0].start, fixings[2].start).value));
    REQUIRE(r.next_discount(fixings[1].start, InterestRate(fwr2), 1.0).to_interest_rate(fixings[1].start.value).value == Catch::Approx(0.1664));

    REQUIRE(r.direct_discount(fixings[0].start, fixings[1].start).to_interest_rate(fixings[1].start.value).value == Catch::Approx(0.08));
    REQUIRE(r.direct_discount(fixings[1].start, fixings[2].start).to_interest_rate(fixings[1].start.value).value == Catch::Approx(0.08));
    REQUIRE(r.direct_discount(fixings[0].start, fixings[2].start).to_interest_rate(fixings[1].start.value).value == Catch::Approx(0.1664));
}

TEST_CASE("POO", "[npv]")
{
    using namespace date;

    auto start_date = jan / day(1) / 2020;
    auto end_date = jan / day(1) / 2030;

    Calendar cal{ start_date, end_date, 12, DayCountConvention::EQUALS };
    InterestRate ir(0.08);

    REQUIRE(StartCashFlow(cal, ir, 1000).to_end_cashflow().cash == Catch::Approx(2158.9249972728));
    REQUIRE(StartCashFlow(cal, ir, 1000).to_coupon().cash == Catch::Approx(149.0294886971));
    REQUIRE(CouponCashFlow(cal, ir, 149.0294886971).to_start_cashflow().cash == Catch::Approx(1000.0));
    REQUIRE(CouponCashFlow(cal, ir, 149.0294886971).to_end_cashflow().cash == Catch::Approx(2158.9249972731));
    REQUIRE(EndCashFlow(cal, ir, 2158.9249972728).to_start_cashflow().cash == Catch::Approx(1000.0));
    REQUIRE(EndCashFlow(cal, ir, 2158.9249972728).to_coupon().cash == Catch::Approx(149.0294886971));
    REQUIRE(CouponCashFlow(cal, ir, 149.0294886971, InterestRate(0.25)).to_start_cashflow().cash == Catch::Approx(2905.0454275324));
    REQUIRE(CouponCashFlow(cal, ir, 149.0294886971, InterestRate(-0.09)).to_start_cashflow().cash == Catch::Approx(718.5193716059));
    REQUIRE(CouponCashFlow(cal, ir, 149.0294886971, InterestRate(0.25)).to_end_cashflow().cash == Catch::Approx(6271.7751917127));
    REQUIRE(CouponCashFlow(cal, ir, 149.0294886971, InterestRate(-0.09)).to_end_cashflow().cash == Catch::Approx(1551.2294323847));

    // crecimiento del dividendo
    auto growth = on_capital(0.20, 0.25, 3, Convention::YIELD, Frequency::QUATERLY);
    REQUIRE(growth.value == Catch::Approx(0.0750770605));
    REQUIRE(CouponCashFlow(cal, InterestRate(0.03), 2000, growth).to_end_cashflow().cash == Catch::Approx(32192.5659896183));
}
