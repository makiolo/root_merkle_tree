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
//#include "rapidcsv.h"

namespace qs {

    using namespace date;

    class DiscountFactor;
    class Maturity;
    class ForwardPeriod;
    class Schedule;
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
    double discount2rate(double df, double maturity, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL);
    double rate2discount(double zc, double maturity, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL);
    InterestRate equivalent_rate(const Maturity& maturity, double rate, Convention conv, int compound_times, Convention other_convention = Convention::YIELD, int other_compound_times = Frequency::ANNUAL);
    InterestRate equivalent_rate(double rate, Convention conv, int compound_times, Convention other_convention = Convention::YIELD, int other_compound_times = Frequency::ANNUAL);
    InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times = Frequency::ANNUAL);

    // one cash flow
    double to_present_value(double cash, const InterestRate& r, const Maturity& maturity);
    double to_future_value(double cash, const InterestRate& r, const Maturity& maturity);
    double to_present_value(double cash, const InterestRate& r, const Schedule& cal);
    double to_future_value(double cash, const InterestRate& r, const Schedule& cal);

    // fv - coupon - fv
    double npv_from_coupon(double coupon, const InterestRate& interest_rate, const Schedule& cal);
    double coupon_from_npv(double npv, const InterestRate& interest_rate, const Schedule& cal);
    double coupon_from_fv(double fv, const InterestRate& interest_rate, const Schedule& cal);
    double fv_from_coupon(double coupon, const InterestRate& interest_rate, const Schedule& cal);

    // growth_coupon
    double npv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Schedule& cal);
    double fv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Schedule& cal);
    double coupon_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Schedule& cal);

    /*
    growth_coupon_from_npv
    growth_coupon_from_coupon
    grouth_coupon_from_fv
    */
    InterestRate compute_par_yield(const std::vector<double>& cf, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

    // value products
    double classic_npv(double investment, double coupon, const InterestRate& interest_rate, const Schedule& cal);
    double bond_npv(double face_value, double coupon, const InterestRate& interest_rate, const Schedule& cal);
    double stock_npv(double investment, double dividend, const InterestRate& interest_rate, const Schedule& cal);


    class Maturity
    {
    public:
        Maturity(double value_)
                : value(value_)
                , has_pillar(false)
        {
            ;
        }

        explicit Maturity(const date::year_month_day& pillar_, double value_)
                : pillar(pillar_)
                , value(value_)
                , has_pillar(true)
        {
            ;
        }

        [[nodiscard]] ForwardPeriod to(const Maturity& target) const;
        [[nodiscard]] DiscountFactor get_discount_factor(const InterestRate& ir) const;

        bool operator>(const Maturity& rhs) const
        {
            return value > rhs.value;
        }

        bool operator<(const Maturity& rhs) const
        {
            return value < rhs.value;
        }

        bool operator>=(const Maturity& rhs) const
        {
            return value >= rhs.value;
        }

        bool operator<=(const Maturity& rhs) const
        {
            return value <= rhs.value;
        }

        bool has_pillar;
        date::year_month_day pillar;
        double value;

        static const Maturity ZERO;
        static const Maturity ONE;
    };

    const Maturity Maturity::ZERO(0.0);
    const Maturity Maturity::ONE(1.0);


    // calcular value
    InterestRate on_capital(double initial_value, double final_value, const Maturity& maturity = Maturity::ONE, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);


    class ForwardPeriod
    {
    public:
        ForwardPeriod(const Maturity& start_, const Maturity& end_)
                : start(start_)
                , end(end_)
        {

        }

        [[nodiscard]] DiscountFactor discount_factor(const InterestRate& ir) const;
        [[nodiscard]] DiscountFactor discount_factor(const InterestRate& ir_start, const InterestRate& ir_end) const;
        [[nodiscard]] InterestRate forward_rate(const InterestRate& ir) const;
        [[nodiscard]] InterestRate forward_rate(const InterestRate& ir_start, const InterestRate& ir_end) const;
        [[nodiscard]] DiscountFactor next_discount_factor(const InterestRate& ir) const;

        Maturity start;
        Maturity end;
    };

    class ZeroPeriod : public ForwardPeriod
    {
    public:
        explicit ZeroPeriod(const Maturity& end)
            : ForwardPeriod(Maturity::ZERO, end)
        {

        }
    };

    [[nodiscard]] ForwardPeriod Maturity::to(const Maturity& target) const
    {
        return {*this, target};
    }

    std::ostream& operator<<(std::ostream& out, const Maturity& obj)
    {
        if (obj.has_pillar)
        {
            out << obj.pillar << " (" << obj.value << ")";
        }
        else
        {
            out << obj.value;
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const ForwardPeriod& obj)
    {
        out << "From period:\t" << obj.start << "\tTo period:\t" << obj.end;
        return out;
    }

    struct Calendar
    {
        // TODO:
    };

    struct Schedule
    {
        date::year_month_day start_date;
        date::year_month_day end_date;
        int tenor;
        DayCountConvention dc_convention;

        Schedule(const date::year_month_day& start_date_, const date::year_month_day& end_date_, int tenor_ = 12, DayCountConvention dc_convention_ = DayCountConvention::EQUALS)
                : start_date(start_date_)
                , end_date(end_date_)
                , tenor(tenor_)
                , dc_convention(dc_convention_)
        {
            build();
        }

        Schedule(const date::year_month_day& start_date_, int duration, int tenor_ = 12, DayCountConvention dc_convention_ = DayCountConvention::EQUALS)
                : start_date(start_date_)
                , end_date(start_date_ + date::years(duration))
                , tenor(tenor_)
                , dc_convention(dc_convention_)
        {
            build();
        }

        Schedule(int duration, int tenor_ = 12, DayCountConvention dc_convention_ = DayCountConvention::EQUALS)
                : start_date(jan / day(1) / 2020)
                , end_date((jan / day(1) / 2020) + date::years(duration))
                , tenor(tenor_)
                , dc_convention(dc_convention_)
        {
            build();
        }

        [[nodiscard]] const std::vector<ForwardPeriod>& get() const
        {
            return forward_periods;
        }

        [[nodiscard]] ForwardPeriod get_first_period() const
        {
            auto mats = forward_periods;
            return mats.front();
        }

        [[nodiscard]] ForwardPeriod get_last_period() const
        {
            auto mats = forward_periods;
            return mats.back();
        }

        [[nodiscard]] std::vector<InterestRate> spot_to_forward(const std::vector<InterestRate>& spots, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL) const;

    protected:
        void build()
        {
            using namespace date;

            std::vector<ForwardPeriod> data;
            auto pillar_day = start_date;
            int i = 0;
            double prev = 0.0, count = 0.0;

            // jump to 1
            pillar_day += months(tenor);
            i++;

            while (pillar_day <= end_date)
            {
                prev = count;
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
                auto start = Maturity(pillar_day, prev);
                pillar_day += months(tenor);
                auto end = Maturity(pillar_day, count);
                forward_periods.emplace_back(start, end);
                spot_periods.emplace_back(end);
                i += 1;
            }
        }

    protected:
        std::vector<ForwardPeriod> forward_periods;
        std::vector<ZeroPeriod> spot_periods;
    };

    class DiscountFactor
    {
    public:
        DiscountFactor(double value_)
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

        [[nodiscard]] InterestRate to_interest_rate(const Maturity& maturity = Maturity::ONE, Convention convention_ = Convention::YIELD, int compound_times_ = Frequency::ANNUAL) const;

        friend DiscountFactor operator*(const DiscountFactor& l, const DiscountFactor& r)
        {
            return {l.value * r.value};
        }

        friend DiscountFactor operator/(const DiscountFactor& l, const DiscountFactor& r)
        {
            return {l.value / r.value};
        }

        friend DiscountFactor operator*(const DiscountFactor& l, double r)
        {
            return {l.value * r};
        }

        friend DiscountFactor operator/(const DiscountFactor& l, double r)
        {
            return {l.value / r};
        }

        friend DiscountFactor operator+(const DiscountFactor& l, const DiscountFactor& r)
        {
            return {l.value + r.value};
        }

        friend DiscountFactor operator-(const DiscountFactor& l, const DiscountFactor& r)
        {
            return {l.value - r.value};
        }

    public:
        double value;
    };

    class InterestRate
    {
    public:
        explicit InterestRate(double value_, Convention convention_ = Convention::YIELD, int compound_times_ = Frequency::ANNUAL)
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

        [[nodiscard]] std::vector<DiscountFactor> get_discount_factor_start(const Schedule& cal) const
        {
            std::vector<DiscountFactor> dfs;
            for (auto& period : cal.get())
            {
                dfs.push_back(period.start.get_discount_factor(*this));
            }
            return dfs;
        }

        [[nodiscard]] std::vector<DiscountFactor> get_discount_factors_end(const Schedule& cal) const
        {
            std::vector<DiscountFactor> dfs;
            for (auto& period : cal.get())
            {
                dfs.push_back(period.end.get_discount_factor(*this));
            }
            return dfs;
        }

        [[nodiscard]] DiscountFactor to_discount_factor(const Maturity& maturity) const
        {
            return DiscountFactor(rate2discount(value, maturity.value, conv, c));
        }

        [[nodiscard]] InterestRate to_other_interest_rate(Convention other_convention, int other_compound_times = Frequency::ANNUAL) const
        {
            return equivalent_rate(value, conv, c, other_convention, other_compound_times);
        }

        [[nodiscard]] InterestRate to_other_interest_rate(const Maturity& maturity, Convention other_convention, int other_compound_times = Frequency::ANNUAL) const
        {
            return equivalent_rate(maturity, value, conv, c, other_convention, other_compound_times);
        }

    public:
        double value;  // annual rate
        int c;  // reinversions each year
        Convention conv;  // convention

        static const InterestRate ZERO;
    };

    const InterestRate InterestRate::ZERO(0.0);

    DiscountFactor ForwardPeriod::discount_factor(const InterestRate& ir) const
    {
        double df0 = ir.to_discount_factor(start).value;
        double df1 = ir.to_discount_factor(end).value;

        return {df1 / df0};
    }

    DiscountFactor ForwardPeriod::discount_factor(const InterestRate& ir_start, const InterestRate& ir_end) const
    {
        double df0 = ir_start.to_discount_factor(start).value;
        double df1 = ir_end.to_discount_factor(end).value;

        return {df1 / df0};
    }

    InterestRate ForwardPeriod::forward_rate(const InterestRate& ir) const
    {
        double df0 = ir.to_discount_factor(start).value;
        double df1 = ir.to_discount_factor(end).value;

        double m = end.value - start.value;

        return InterestRate((df0 / df1 - 1.0) / m);
    }

    InterestRate ForwardPeriod::forward_rate(const InterestRate& ir_start, const InterestRate& ir_end) const
    {
        double df0 = ir_start.to_discount_factor(start).value;
        double df1 = ir_end.to_discount_factor(end).value;

        double m = end.value - start.value;

        return InterestRate((df0 / df1 - 1.0) / m);
    }

    DiscountFactor ForwardPeriod::next_discount_factor(const InterestRate& ir) const
    {
        double discount = ir.to_discount_factor(end).value;
        double m = end.value - start.value;
        return {discount / (1.0 + m * forward_rate(ir).value)};
    }

    [[nodiscard]] DiscountFactor Maturity::get_discount_factor(const InterestRate& ir) const
    {
        return ir.to_discount_factor(*this);
    }

    std::vector<InterestRate> Schedule::spot_to_forward(const std::vector<InterestRate>& spots, Convention conv, int compound_times) const
    {
        std::vector<InterestRate> fwds;

        int i = 0;
        for(auto& period : get())
        {
            if(i == 0)
            {
                fwds.push_back(spots[i]);
            }
            else
            {
                InterestRate fwd = period.discount_factor(spots[i-1], spots[i]).to_interest_rate(Maturity::ONE, conv, compound_times);
                fwds.push_back(fwd);
            }
            i += 1;
        }
        return fwds;
    }

    //////////////////////////////////////////////////////////////

    struct Engine
    {
        // TODO: Montecarlo Engine
    };

    struct Model
    {
        // TODO: Brownian model
    };

    class Leg;

    struct Product
    {
        Schedule cal;
        std::vector<Leg> legs;
    };

    class CashFlow  // Request of Transaction
    {
    public:
        explicit CashFlow(const Schedule& cal_, const InterestRate& ir_, double cash_)
                : cal(cal_)
                , ir(ir_)
                , cash(cash_)
        {
            ;
        }

    public:
        Schedule cal;
        InterestRate ir;
        double cash;
    };

    class Leg
    {
        std::vector<CashFlow> flows;
    };

    class CouponCashFlow : public CashFlow
    {
    public:
        CouponCashFlow(const Schedule& cal_, const InterestRate& ir_, double cash_, const InterestRate& growth_ = InterestRate::ZERO)
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
        StartCashFlow(const Schedule& cal_, const InterestRate& ir_, double cash_)
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
        EndCashFlow(const Schedule& cal_, const InterestRate& ir_, double cash_)
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
        CustomCashFlow(const Schedule& cal_, const InterestRate& ir_, double cash_, const Maturity& maturity_)
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

        // to other custom cashflow ?

    protected:
        Maturity maturity;
    };

    InterestRate DiscountFactor::to_interest_rate(const Maturity& maturity, Convention convention_, int compound_times_) const
    {
        return InterestRate(discount2rate(value, maturity.value, convention_, compound_times_), convention_, compound_times_);
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

    double discount2rate(double df, double maturity, Convention conv, int compound_times)
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

    double rate2discount(double zc, double maturity, Convention conv, int compound_times)
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

    InterestRate compute_par_yield(const std::vector<double>& cf, Convention convention, int compound_times)
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
                discount_factor = rate2discount(guessRate, j, convention, compound_times);
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
        return InterestRate(guessRate, convention, compound_times);
    }

    InterestRate equivalent_rate(const Maturity& maturity, double rate, Convention convention, int compound_times, Convention other_convention, int other_compound_times)
    {
        return InterestRate(rate, convention, compound_times)
                .to_discount_factor(maturity)
                .to_interest_rate(maturity, other_convention, other_compound_times);
    }

    InterestRate equivalent_rate(double rate, Convention convention, int compound_times, Convention other_convention, int other_compound_times)
    {
        return InterestRate(rate, convention, compound_times)
                .to_discount_factor(Maturity::ONE)
                .to_interest_rate(Maturity::ONE, other_convention, other_compound_times);
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
    double to_present_value(double cash, const InterestRate& r, const Schedule& cal)
    {
        auto maturity = cal.get_last_period().end;  // obtener maturity del cash (teniendo cierto "cal")
        return to_present_value(cash, r, maturity);
    }

    // tenemos un en 0 y nos lo traemos al final del calendario
    double to_future_value(double cash, const InterestRate& r, const Schedule& cal)
    {
        auto maturity = cal.get_last_period().end;  // obtener maturity del cash (teniendo cierto "cal")
        return to_future_value(cash, r, maturity);
    }

    // only coupons
    double npv_from_coupon(double coupon, const InterestRate& interest_rate, const Schedule& cal)
    {
        return npv_from_growth_coupon(coupon, InterestRate::ZERO, interest_rate, cal);
    }

    double npv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Schedule& cal)
    {
        double npv = 0.0;
        double i = 0.0;
        for (const auto& df : interest_rate.get_discount_factors_end(cal))
        {
            // TODO: revisar ese i ?, que maturity es esa?
            npv += (df.value * (coupon / interest_rate.c)) / growth_rate.to_discount_factor(i).value;
            i += 1.0;
        }
        return npv;
    }

    double fv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Schedule& cal)
    {
        double npv = npv_from_growth_coupon(coupon, growth_rate, interest_rate, cal);
        return to_future_value(npv, interest_rate, cal);
    }

    double fv_from_coupon(double coupon, const InterestRate& interest_rate, const Schedule& cal)
    {
        return fv_from_growth_coupon(coupon, InterestRate(0.0), interest_rate, cal);
    }

    double coupon_from_npv(double npv, const InterestRate& interest_rate, const Schedule& cal)
    {
        double total_df = 0.0;
        for (const auto& df : interest_rate.get_discount_factors_end(cal))
        {
            total_df += df.value;
        }
        return (npv * interest_rate.c) / total_df;
    }

    // VAN = coupons - initial investement
    double classic_npv(double investment, double coupon, const InterestRate& interest_rate, const Schedule& cal)
    {
        return CouponCashFlow(cal, interest_rate, coupon).to_start_cashflow().cash - investment;
    }

    // coupons + payment on yield-maturity
    double bond_npv(double face_value, double coupon, const InterestRate& interest_rate, const Schedule& cal)
    {
        double coupons = CouponCashFlow(cal, interest_rate, coupon).to_start_cashflow().cash;
        double yield_on_payment = EndCashFlow(cal, interest_rate, face_value).to_start_cashflow().cash;
        return coupons + yield_on_payment;
    }

    // stock investment = coupons + payment on yield-maturity - initial investement
    double stock_npv(double investment, double dividend, const InterestRate& interest_rate, const Schedule& cal)
    {
        double coupon = investment * (dividend - interest_rate.value);
        double dividends = CouponCashFlow(cal, interest_rate, coupon).to_start_cashflow().cash;
        double yield_on_payment = EndCashFlow(cal, interest_rate, investment).to_start_cashflow().cash;
        double npv = dividends + yield_on_payment - investment;
        return npv;
    }

    double coupon_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Schedule& cal)
    {
        double npv = npv_from_growth_coupon(coupon, growth_rate, interest_rate, cal);
        return coupon_from_npv(npv, interest_rate, cal);
    }

    double coupon_from_fv(double fv, const InterestRate& interest_rate, const Schedule& cal)
    {
        double total_df = 0.0;
        for (const auto& df : interest_rate.get_discount_factor_start(cal))
        {
            total_df += 1.0 / df.value;
        }
        return (fv * interest_rate.c) / total_df;
    }

    InterestRate on_capital(double initial_value, double final_value, const Maturity& maturity, Convention convention, int compound_times)
    {
        if (convention == Convention::YIELD && compound_times == Frequency::ANNUAL)
        {
            // cagr
            return InterestRate(pow(final_value / initial_value, 1.0 / maturity.value) - 1.0);
        }
        else
        {
            return InterestRate((final_value - initial_value) / initial_value, Convention::LINEAR)
                    .to_discount_factor(Maturity::ONE)
                    .to_interest_rate(maturity, convention, compound_times);
        }
    }
    
    std::vector<InterestRate> par_to_spot(const std::vector<InterestRate>& pares, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL)
    {
        // bootstraping
        std::vector<InterestRate> spots;
        double tenor_inc = 1.0;
        double tenor = 0.0;
        bool first = true;
        for(auto& ir : pares)
        {
            if(first)
            {
                spots.push_back(ir);
                first = false;
            }
            else
            {
                double accum = 0.0;
                for(int i=0; i < tenor; ++i)
                {
                    accum += to_present_value(ir.value, spots[i], Maturity(i + 1));
                }
                InterestRate spotN = DiscountFactor((1.0 - accum) / (1.0 + ir.value))
                                        .to_interest_rate(tenor + tenor_inc, conv, compound_times);
                spots.push_back(spotN);
            }
            tenor += tenor_inc;
        }
        return spots;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////

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
                Schedule(2022_y / 1 / 1, 20));

        REQUIRE(npv == Catch::Approx(6135.87));
    }

    TEST_CASE("fv_from_coupon", "[fv]") {

        using namespace date;

        // Ahorro inicial en el futuro

        double initial = 10000;
        double r = 0.07;
        int maturity = 8;
        auto cal = Schedule(2022_y / 1 / 1, maturity);
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
        auto cal2 = Schedule(2022_y / 1 / 1, maturity, 1);
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
        auto cal = Schedule(2022_y / 1 / 1, maturity);
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
        REQUIRE(to_future_value(final, InterestRate(r, Convention::YIELD, 12), Maturity(forward_years)) == Catch::Approx(44524.0670913586));

        // trading
        initial = 5000;
        r = 0.10;
        double r_anual = equivalent_rate(r, 12, 1).value;
        double years = 3.0;
        REQUIRE(to_future_value(initial, InterestRate(r, Convention::YIELD, 12), Maturity(years)) == \
            Catch::Approx(to_future_value(initial, InterestRate(r_anual), Maturity(years))));
    }

    TEST_CASE("value & zc", "[fv]") {

        REQUIRE(rate2discount(discount2rate(0.95, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.95));
        REQUIRE(discount2rate(rate2discount(0.05, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.05));

        REQUIRE(rate2discount(discount2rate(0.95, 3, Convention::LINEAR, 4), 3, Convention::LINEAR) == Catch::Approx(0.95));
        REQUIRE(discount2rate(rate2discount(0.05, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.05));

        REQUIRE(rate2discount(discount2rate(0.95, 3, Convention::YIELD), 3, Convention::YIELD) == Catch::Approx(0.95));
        REQUIRE(discount2rate(rate2discount(0.05, 3, Convention::YIELD), 3, Convention::YIELD) == Catch::Approx(0.05));

        REQUIRE(rate2discount(discount2rate(0.95, 3, Convention::YIELD, 4), 3, Convention::YIELD, 4) == Catch::Approx(0.95));
        REQUIRE(discount2rate(rate2discount(0.05, 3, Convention::YIELD, 4), 3, Convention::YIELD, 4) == Catch::Approx(0.05));

        REQUIRE(rate2discount(discount2rate(0.95, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.95));
        REQUIRE(discount2rate(rate2discount(0.05, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.05));

        REQUIRE(rate2discount(discount2rate(0.95, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.95));
        REQUIRE(discount2rate(rate2discount(0.05, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.05));
    }

    TEST_CASE("bond_npv2", "[fv]") {

        using namespace date;

        double cash = 17181.8617983192;
        double r = 0.07;
        int maturity = 8;
        auto cal = Schedule(2022_y / 1 / 1, maturity);
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
                Schedule(2022_y / 1 / 1, 1)) == Catch::Approx(-5504.9504950495));

        double npv1 = classic_npv(1000, 100, InterestRate(-0.1940185202), Schedule(2022_y / 1 / 1, 6));
        REQUIRE(npv1 == Catch::Approx(364.7956282082));

        std::vector<double> cf = { -1000, 100, 100, 100, 100, 100 };
        InterestRate irr = compute_par_yield(cf);
        REQUIRE(irr.value == Catch::Approx(-0.1940185202));
    }

    TEST_CASE("real coupon", "[fv]") {

        using namespace date;

        double coupon_netflix = 9.9;
        int maturity = 10;

        auto cal = Schedule(2022_y / 1 / 1, maturity);

        double npv = stock_npv(1000, 0.08, InterestRate(0.03), cal);
        REQUIRE(npv == Catch::Approx(170.6040567355));

        double npv_ = stock_npv(1000, 0.08, InterestRate(0.12), cal);
        REQUIRE(npv_ == Catch::Approx(-904.0356845457));
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
        double  fv = to_future_value(1000, InterestRate(0.10 * 12, Convention::YIELD, 12), Maturity::ONE);
        REQUIRE(fv == Catch::Approx(3138.428376721));
        REQUIRE(on_capital(1000, fv).value == Catch::Approx(equivalent_rate(0.10 * 12, 12, 1).value));

        /*
        10% mensual con reinversion anual = 120%
        */
        double  fv2 = to_future_value(1000, InterestRate(0.10 * 12), Maturity::ONE);
        REQUIRE(fv2 == Catch::Approx(2200.0));
        REQUIRE(on_capital(1000, fv2).value == Catch::Approx(equivalent_rate(0.10 * 12, 1, 1).value));

        /*
        2% semanal con reinversion semanal = 191.34%
        */
        double  fv3 = to_future_value(1000, InterestRate(0.02 * 54, Convention::YIELD, 54), Maturity::ONE);
        REQUIRE(fv3 == Catch::Approx(2913.4614441403));
        REQUIRE(on_capital(1000, fv3).value == Catch::Approx(InterestRate(0.02 * 54, Convention::YIELD, 54).to_other_interest_rate(Convention::YIELD).value));

        /*
        2% semanal con reinversion continua = 194.46%
        */
        double fv4 = to_future_value(1000, InterestRate(0.02 * 54, Convention::EXPONENTIAL), Maturity::ONE);
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

        auto cal = Schedule(2022_y / 1 / 1, 5);

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

        auto cal = Schedule(2022_y / 1 / 1, 5, 1);

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
            std::cout << "for: " << d << " (dia " << d.day() << ")" << ": " << to_present_value(1000, InterestRate(0.05), Maturity(maturity)) << std::endl;
            last_maturity = maturity;
        }
        REQUIRE(last_maturity == Catch::Approx(10.0082191781));

        Schedule cal{ start_date, end_date, 3, DayCountConvention::EQUALS };

        std::vector<double> v1, v2;
        for (auto& period : cal.get())
        {
            std::cout << "begin mode = true, pillar: " << period.start.pillar << " - value: " << period.start.value << std::endl;
            v1.push_back(period.start.value);
        }
        for (auto& period : cal.get())
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

        Schedule cal{ start_date, end_date, 12, DayCountConvention::EQUALS };
        auto fixings = cal.get();

        InterestRate r(0.08);

        double cash = 1;

        double fwd2 = fixings[0].discount_factor(r).value;
        double fwd3 = fixings[1].discount_factor(r).value;
        double fwd4 = fixings[0].start.to(fixings[2].start).discount_factor(r).value;

        REQUIRE((fwd2) == Catch::Approx(0.9259259259));
        REQUIRE((fwd3) == Catch::Approx(0.9259259259));
        REQUIRE((fwd4) == Catch::Approx(0.9259259259 * 0.9259259259));

        double fwr1 = fixings[0].forward_rate(r).to_other_interest_rate(Convention::YIELD, 12).value / 12.0;
        double fwr2 = fixings[1].forward_rate(r).value;
        double fwr3 = fixings[2].forward_rate(r).value;

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

        //
        REQUIRE(fixings[0].start.to(fixings[1].start).discount_factor(r).to_interest_rate(fixings[1].start).value == Catch::Approx(0.08));
        REQUIRE(fixings[0].discount_factor(r).to_interest_rate(fixings[1].start).value == Catch::Approx(0.08));
        REQUIRE(fixings[0].start.to(fixings[1].start).next_discount_factor(r).to_interest_rate(fixings[1].start).value == Catch::Approx(0.1664)); // ??
        //
        REQUIRE(fixings[1].discount_factor(r).to_interest_rate(fixings[1].start).value == Catch::Approx(0.08));
        REQUIRE(fixings[0].start.to(fixings[2].start).discount_factor(r).to_interest_rate(fixings[1].start).value == Catch::Approx(0.1664)); // ??
        //
        REQUIRE(fixings[2].forward_rate(r).value == fixings[2].start.to(fixings[2].end).forward_rate(r).value);
        //
        REQUIRE(Catch::Approx(0.09012224) == fixings[1].start.to(fixings[4].end).forward_rate(r).value);
        REQUIRE(Catch::Approx(0.0938656154) == fixings[0].start.to(fixings[4].end).forward_rate(r).value);
        //
        REQUIRE(Catch::Approx(0.7350298528) == fixings[1].start.to(fixings[4].end).discount_factor(r).value);
        REQUIRE(Catch::Approx(0.680583197) == fixings[0].start.to(fixings[4].end).discount_factor(r).value);
        //
        REQUIRE(fixings[0].start.to(fixings[1].start).next_discount_factor(r).value == Catch::Approx(fixings[0].start.to(fixings[2].start).discount_factor(r).value));
        REQUIRE(fixings[0].start.to(fixings[2].start).next_discount_factor(r).value == Catch::Approx(fixings[0].start.to(fixings[4].start).discount_factor(r).value));
        REQUIRE(fixings[0].start.to(fixings[3].start).next_discount_factor(r).value == Catch::Approx(fixings[0].start.to(fixings[6].start).discount_factor(r).value));
    }

    TEST_CASE("POO", "[npv]")
    {
        using namespace date;

        auto start_date = jan / day(1) / 2020;
        auto end_date = jan / day(1) / 2030;

        Schedule cal{ start_date, end_date, 12, DayCountConvention::EQUALS };
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

    TEST_CASE("forwards", "[fr]")
    {
        using namespace date;

        auto start_date = jan / day(1) / 2020;
        auto end_date = jan / day(1) / 2030;

        Schedule cal{ start_date, end_date, 6, DayCountConvention::EQUALS };

        InterestRate r(0.12);
        auto periods = cal.get();

        for (auto& period : periods)
        {
            std::cout << "start: " << period.start.value << ", end: " << period.end.value << " - fr: " << period.forward_rate(r).value << std::endl;
        }
    }

    TEST_CASE("csv", "[csv]")
    {
        /*
        rapidcsv::Document doc(R"(D:\dev\deeplearning_dev\dataset\correlation.csv)");

        std::vector<float> col = doc.GetColumn<float>("EURGBP_Close");
        for(const auto& price : col)
        {
            std::cout << price << std::endl;
        }
        std::cout << "Read " << col.size() << " values." << std::endl;
        */
    }

    TEST_CASE("spot rate", "[ir]")
    {
        InterestRate ir = on_capital(8900, 10000, 5);
        REQUIRE(ir.value == Catch::Approx(0.0235804883));
    }

    TEST_CASE("curve yield", "[ir]")
    {
        /*
Tenor	Type	Frequency	Daycount	SwapRate	Date	YearFraction	CumYearFraction	ZeroRate
ON	Deposit	Zero Coupon	ACT360	0.0003	2021-04-21	0.002777778	0.002777778	0.0003
1W	Deposit	Zero Coupon	ACT360	0.000724	2021-04-27	0.016666667	0.019444444	0.000723995
1M	Deposit	Zero Coupon	ACT360	0.001185	2021-05-20	0.063888889	0.083333333	0.001184941
3M	Deposit	Zero Coupon	ACT360	0.0018838	2021-07-20	0.169444444	0.252777778	0.001883352
6M	Deposit	Zero Coupon	ACT360	0.00203	2021-10-20	0.255555556	0.508333333	0.002028953
9M	Deposit	Zero Coupon	ACT360	0.0025	2022-01-20	0.255555556	0.763888889	0.002497616
1Y	Deposit	Zero Coupon	ACT360	0.0028375	2022-04-20	0.25	1.013888889	0.002833426
15M	EuroDollarFuture	Q	ACT360	0.00318125	2022-07-20	0.252777778	1.266666667	0.002909634
18M	EuroDollarFuture	Q	ACT360	0.003525	2022-10-20	0.255555556	1.522222222	0.003025822
21M	EuroDollarFuture	Q	ACT360	0.003583	2023-01-20	0.255555556	1.777777778	0.003117122
2Y	EuroDollarFuture	Q	ACT360	0.003641	2023-04-20	0.25	2.027777778	0.003181505
3Y	EuroDollarFuture	Q	ACT360	0.003797	2024-04-22	1.022222222	3.05	0.007308586
4Y	EuroDollarFuture	Q	ACT360	0.004	2025-04-21	1.011111111	4.061111111	0.009508661
5Y	Swap	S	ACT360	0.00823	2026-04-20	1.011111111	5.072222222	0.008125758
7Y	Swap	S	ACT360	0.01155	2028-04-20	2.030555556	7.102777778	0.011558032
10Y	Swap	S	ACT360	0.01468	2031-04-21	3.044444444	10.14722222	0.013906471
15Y	Swap	S	ACT360	0.01729	2036-04-21	5.075	15.22222222	0.023043104
30Y	Swap	S	ACT360	0.01877	2051-04-20	15.21388889	30.43611111	0.011799239


Leg1			Leg2			Price
FxdFlt	Fixed		FxdFlt	Float		Leg1PV	139013.01
RateSpread	0.004		RateSpread	0		Leg2PV	-138815.85
Leg1SettlementDate	15-Apr-21		Leg2SettlementDate	15-Apr-21		SwapNPV	197.17
Leg1Frequency	6M		Leg2Frequency	3M
Leg1Horizon	4Y		Leg2Horizon	4Y
Leg1Calendar	NY		Leg2BusinessCalendar	NY
Leg1BusinessDayConvention	Following		Leg2BusinessDayConvention	Following
Leg1DayCountConvention	Thirty360		Leg2DayCountConvention	ACT365
FxdNotional	10000000		FltNotional	-10000000

         */

        // Deposits
        // https://www.euribor-rates.eu/en/current-euribor-rates/
        double cumyear0 = 0.002777778;
        double swaprate0 = 0.0003;
        double zerorate0 = (1.0 / cumyear0) * log(1.0 + swaprate0 * cumyear0);
        REQUIRE(zerorate0 == Catch::Approx(0.0003));
        REQUIRE(zerorate0 == Catch::Approx(InterestRate(swaprate0, Convention::LINEAR).to_other_interest_rate(cumyear0, Convention::EXPONENTIAL).value));

        double cumyear1 = 0.019444444;
        double swaprate1 = 0.000724;
        double zerorate1 = (1.0 / cumyear1) * log(1.0 + swaprate1 * cumyear1);
        REQUIRE(zerorate1 == Catch::Approx(0.000723995));
        REQUIRE(zerorate1 == Catch::Approx(InterestRate(swaprate1, Convention::LINEAR).to_other_interest_rate(cumyear1, Convention::EXPONENTIAL).value));

        double cumyear_1y = 1.013888889;
        double swaprate_1y = 0.0028375;
        double zerorate_1y = (1.0 / cumyear_1y) * log(1.0 + swaprate_1y * cumyear_1y);
        REQUIRE(zerorate_1y == Catch::Approx(0.002833426));
        REQUIRE(zerorate_1y == Catch::Approx(InterestRate(swaprate_1y, Convention::LINEAR).to_other_interest_rate(cumyear_1y, Convention::EXPONENTIAL).value));

        // Futures
        // https://www.cmegroup.com/markets/interest-rates/stirs/eurodollar.contractSpecs.html
        double zerorate_2y_prev = 0.003117122;
        double cumyear_2y_prev = 1.777777778;
        double dcf_2y = 0.25;
        double cumyear_2y = 2.027777778;
        double swaprate_2y = 0.003641;
        /*
        R_eurofut_continuous = 4 × ln[(1 + R_quarterly × YF)]
        R_continuous = (R_eurofut_continuous × YF + R_continuous_t-1 × CYF_t-1) / CYF
        */
        double r_continuous_2y = 4.0 * log(1 + swaprate_2y * dcf_2y);
        REQUIRE(r_continuous_2y == Catch::Approx(InterestRate(swaprate_2y, Convention::YIELD, Frequency::QUATERLY).to_other_interest_rate(cumyear_2y, Convention::EXPONENTIAL).value));
        REQUIRE(r_continuous_2y == Catch::Approx(0.0036393439));

        // ForwardPeriod per = ForwardPeriod(Maturity(cumyear_2y_prev), Maturity(cumyear_2y));
        // InterestRate fr = per.forward_rate(InterestRate(zerorate_2y_prev, Convention::EXPONENTIAL));
        // REQUIRE(fr.value == Catch::Approx(0.003181505));

        // DiscountFactor df = InterestRate(r_continuous_2y, Convention::EXPONENTIAL).to_discount_factor(dcf_2y) / InterestRate(zerorate_2y_prev, Convention::EXPONENTIAL).to_discount_factor(cumyear_2y_prev);
        // REQUIRE(df.to_interest_rate(Convention::EXPONENTIAL).value == Catch::Approx(0.003181505));

        // apply transformation
        double w1 = cumyear_2y_prev / cumyear_2y;  // 0.8767
        double w2 = dcf_2y / cumyear_2y;           // 0.1232
        double zerorate_2y = (zerorate_2y_prev * w1 + r_continuous_2y * w2);
        REQUIRE(zerorate_2y == Catch::Approx(0.003181505));

        // Swaps
        double dcf_10y = 3.044444444;
        double zerorate_10y_prev = 0.011558032;
        double cumyear_10y_prev = 7.102777778;
        double cumyear_10y = 10.14722222;
        double swaprate_10y = 0.01468;

        double sumproduct = 0.0;
        /*
Tenor	Type	Frequency	Daycount	SwapRate	Date	YearFraction	CumYearFraction	ZeroRate
5Y	Swap	S	ACT360	0.00823	2026-04-20	1.011111111	5.072222222	0.008125758
7Y	Swap	S	ACT360	0.01155	2028-04-20	2.030555556	7.102777778	0.011558032
10Y	Swap	S	ACT360	0.01468	2031-04-21	3.044444444	10.14722222	0.013906471
15Y	Swap	S	ACT360	0.01729	2036-04-21	5.075	15.22222222	0.023043104
30Y	Swap	S	ACT360	0.01877	2051-04-20	15.21388889	30.43611111	0.011799239
        */
        sumproduct += 0.00823 * exp(-0.008125758 * 1.011111111);
        sumproduct += 0.01155 * exp(-0.011558032 * 2.030555556);

        // 0.014626386588449164
        // double r_continuous_10y = InterestRate(swaprate_10y, Convention::YIELD, Frequency::SEMIANNUAL).to_other_interest_rate(cumyear_10y, Convention::EXPONENTIAL).value;

        double zerorate_10y = -1.0 * log((1.0 - sumproduct) / (1.0 + (swaprate_10y / 2.0))) / cumyear_10y;

        // REQUIRE(zerorate_10y == Catch::Approx(0.013906471));
        // REQUIRE(zerorate_10y == Catch::Approx(InterestRate(swaprate_10y, Convention::LINEAR).to_discount_factor(Maturity(cumyear_10y)).to_interest_rate(cumyear_10y, Convention::EXPONENTIAL).value));
        //
        // 0.014626386588449164

    }

    TEST_CASE("property multiply continuous interest", "[ir exp]")
    {
        // add forward_periods

        InterestRate r1 = on_capital(1000, 2000, Maturity(3), Convention::EXPONENTIAL);
        InterestRate r2 = on_capital(2000, 3000, Maturity(1), Convention::EXPONENTIAL);

        InterestRate result = on_capital(1000, 3000, Maturity(4), Convention::EXPONENTIAL);

        REQUIRE(r1.value == Catch::Approx(0.2310490602));
        REQUIRE(r2.value == Catch::Approx(0.4054651081));
        REQUIRE(result.value == Catch::Approx(0.2746530722));

        double w1 = 3.0 / 4.0;
        double w2 = 1.0 / 4.0;
        REQUIRE(result.value == Catch::Approx(r1.value * w1 + r2.value * w2));
    }

    TEST_CASE("property multiply continuous interest with discount factors", "[ir exp]")
    {
        // add forward_periods

        DiscountFactor r1 = on_capital(1000, 2000, Maturity(3), Convention::EXPONENTIAL).to_discount_factor(Maturity(3));
        DiscountFactor r2 = on_capital(2000, 3000, Maturity(1), Convention::EXPONENTIAL).to_discount_factor(Maturity(1));

        DiscountFactor result = on_capital(1000, 3000, Maturity(4), Convention::EXPONENTIAL).to_discount_factor(Maturity(4));

        REQUIRE(r1.value == Catch::Approx(0.5));
        REQUIRE(r2.value == Catch::Approx(0.6666666667));
        REQUIRE(result.value == Catch::Approx(0.3333333333));

        REQUIRE(result.value == Catch::Approx(r1.value * r2.value));
    }

    TEST_CASE("property multiply continuous interest 2", "[ir yield]")
    {
        // add forward_periods
        InterestRate r1 = on_capital(1000, 1100, Maturity(3), Convention::LINEAR).to_other_interest_rate(Maturity(3), Convention::EXPONENTIAL);
        InterestRate r2 = on_capital(1100, 1200, Maturity(1), Convention::YIELD).to_other_interest_rate(Maturity(1), Convention::EXPONENTIAL);

        InterestRate result = on_capital(1000, 1200, Maturity(4), Convention::YIELD).to_other_interest_rate(Maturity(4), Convention::EXPONENTIAL);

        REQUIRE(r1.value == Catch::Approx(0.0317700599));
        REQUIRE(r2.value == Catch::Approx(0.087011377));
        REQUIRE(result.value == Catch::Approx(0.0455803892));

        double w1 = 3.0 / 4.0;
        double w2 = 1.0 / 4.0;
        REQUIRE(result.value == Catch::Approx(r1.value * w1 + r2.value * w2));

        auto start_date = jan / day(1) / 2020;
        auto end_date = jan / day(1) / 2024;

        Schedule cal{ start_date, end_date, 12, DayCountConvention::EQUALS };

        for (auto& period : cal.get())
        {
            std::cout << period << std::endl;
        }

        REQUIRE(StartCashFlow(cal, InterestRate(result.value), 1000).to_end_cashflow().cash == Catch::Approx(1195.1700905152));

    }

    TEST_CASE("bootstraping spot rates from par rates", "[spot rates]")
    {
        auto start_date = jan / day(1) / 2020;
        auto end_date = jan / day(1) / 2023;

        Schedule cal( start_date, end_date );

        // par_rates_to_spot_rates
        // spot_rates_to_forward_rates

        // https://wiki.treasurers.org/wiki/Converting_from_par_rates
        //
        std::vector<InterestRate> pares = {InterestRate(0.04, Convention::YIELD, Frequency::ANNUAL),
                                           InterestRate(0.052, Convention::YIELD, Frequency::ANNUAL),
                                           InterestRate(0.064, Convention::YIELD, Frequency::ANNUAL),
        };
        auto spots = par_to_spot(pares, Convention::YIELD);

        // http://financialexamhelp123.com/par-curve-spot-curve-and-forward-curve/
        // http://financialexamhelp123.com/calculating-forward-rates-from-spot-rates/
        // https://wiki.treasurers.org/wiki/Converting_from_zero_coupon_rates

        auto fwds = cal.spot_to_forward(spots, Convention::YIELD);

        REQUIRE(1 + spots[0].value == Catch::Approx(1 + fwds[0].value));
        REQUIRE(pow(1 + spots[1].value, 2) == Catch::Approx((1+fwds[0].value) * (1+fwds[1].value)));
        REQUIRE(pow(1 + spots[2].value, 3) == Catch::Approx((1+fwds[0].value) * (1+fwds[1].value) * (1+fwds[2].value)));

        // strips structure
        // TODO convert bond prices to DiscountFactors objects
        std::vector< std::pair<long, long> > structure_term;
        structure_term.emplace_back(1, 99);
        structure_term.emplace_back(2, 97);
        structure_term.emplace_back(3, 95);
        structure_term.emplace_back(5, 90);
        structure_term.emplace_back(10, 73);

        std::vector<InterestRate> spots2;

        for(const auto& [maturity, bond_value] : structure_term)
        {
            DiscountFactor bond1(bond_value / 100.0);
            spots2.emplace_back(bond1.to_interest_rate(Maturity(maturity)));
        }

        REQUIRE(spots2[0].value == Catch::Approx(0.0101010101));
        REQUIRE(spots2[1].value == Catch::Approx(0.0153461651));
        REQUIRE(spots2[2].value == Catch::Approx(0.0172447682));
        REQUIRE(spots2[3].value == Catch::Approx(0.0212956876));
        REQUIRE(spots2[4].value == Catch::Approx(0.0319715249));

        for(const auto& spot : spots2)
        {
            double v = spot.value;
            std::cout << v << std::endl;
        }

        std::cout << "-----" << std::endl;

        Schedule cal2(5);

        auto fwds2 = cal2.spot_to_forward(spots2);
        for(const auto& fwd : fwds2)
        {
            double v = fwd.value;
            std::cout << v << std::endl;
        }

        Schedule cal3(4);
        Schedule subcal3(2);

        // interpolation example
        InterestRate ir1(0.05);
        InterestRate ir2(0.07);
        StartCashFlow start_cash1(cal3, ir2, 15000);
        EndCashFlow end_cash = start_cash1.to_end_cashflow();
        std::cout << end_cash.cash << std::endl;

        EndCashFlow end_cash2(subcal3, ir1, 20000);
        StartCashFlow start_cash2 = end_cash2.to_start_cashflow();
        std::cout << start_cash2.cash << std::endl;

        DiscountFactor y2(start_cash2.cash / end_cash2.cash);
        DiscountFactor y4(start_cash1.cash / end_cash.cash);
        std::cout << y2.value << std::endl;
        std::cout << y4.value << std::endl;
        DiscountFactor y3 = (y2 + y4) / 2.0;
        std::cout << y3.value << std::endl;

        std::vector<InterestRate> spots3;
        spots3.push_back(y2.to_interest_rate(Maturity(1)));
        spots3.push_back(y2.to_interest_rate(Maturity(2)));
        spots3.push_back(y3.to_interest_rate(Maturity(3)));
        spots3.push_back(y4.to_interest_rate(Maturity(4)));
        std::cout << "--------" << std::endl;
        for(const auto& spot : spots3)
        {
            double v = spot.value;
            std::cout << v << std::endl;
        }
        std::cout << "--------" << std::endl;
        for(const auto& spot : cal3.spot_to_forward(spots3))
        {
            double v = spot.value;
            std::cout << v << std::endl;
        }
    }
}
