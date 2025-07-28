import { Trans, useLingui } from '@lingui/react/macro';

interface WarningsRecommendationsProps {
  warnings: string[];
  recommendations: string[];
}

export default function WarningsRecommendations({
  warnings,
  recommendations
}: WarningsRecommendationsProps) {
  const { t } = useLingui();
  return (
    <div className='flex gap-[0.5rem]'>
      <div className='area w-full gap-[0.5rem]'>
        <h3 className='subtitle text-[var(--warning-color)]'>
          <Trans>Warnings</Trans>
        </h3>
        <div className='flex h-[11rem] flex-col overflow-y-auto rounded-[0.4rem] border-1 border-[var(--border-color)] p-[0.1rem]'>
          {warnings.length > 0 ? (
            warnings.map((warn, i) => (
              <div key={i} className='warningItem'>
                <p>{warn}</p>
              </div>
            ))
          ) : (
            <div className='warningItem'>
              <p>
                <Trans>No warnings</Trans>
              </p>
            </div>
          )}
        </div>
      </div>

      <div className='area w-full gap-[0.5rem]'>
        <h3 className='subtitle text-primary'>
          <Trans>Recommendations</Trans>
        </h3>
        <div className='bg-card-background flex h-[11rem] flex-col overflow-y-auto rounded-[0.4rem] border-1 border-[var(--border-color)] p-[0.1rem]'>
          {recommendations.length > 0 ? (
            recommendations.map((rec, i) => (
              <div key={i} className='warningItem'>
                <p>{rec}</p>
              </div>
            ))
          ) : (
            <div className='warningItem'>
              <p>
                <Trans>No recommendations</Trans>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
