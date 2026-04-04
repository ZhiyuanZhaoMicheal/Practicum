<template>
  <div class="detail-page">
    <!-- Sticky sidebar TOC -->
    <nav class="detail-toc" v-if="sectionData && subSections.length">
      <RouterLink to="/docs" class="back-link" style="margin-bottom:16px">
        <span class="back-arrow">←</span> All Docs
      </RouterLink>
      <div class="detail-toc__title">{{ sectionData.title }}</div>
      <ul class="detail-toc__list">
        <li v-for="s in subSections" :key="s.id">
          <a
            class="detail-toc__link"
            :class="{ active: activeTocId === s.id }"
            @click.prevent="tocScrollTo(s.id)"
            href="#"
          >{{ s.label }}</a>
        </li>
      </ul>
    </nav>

    <div class="detail-inner">
      <!-- Back link (only if no sidebar) -->
      <RouterLink v-if="!subSections.length" to="/docs" class="back-link">
        <span class="back-arrow">←</span> Back to Documentation
      </RouterLink>

      <!-- Dynamic section content -->
      <article class="detail-content" v-if="sectionData">
        <div class="detail-header reveal">
          <span class="mono dim">{{ sectionData.num }}</span>
          <h1>{{ sectionData.title }}</h1>
          <span v-for="t in sectionData.tags" :key="t" class="tag tag--cyan" style="font-size:10px">{{ t }}</span>
        </div>

        <!-- 01 Overview -->
        <template v-if="sectionId === 'overview'">
          <h2 id="sec-1-1">1.1 The Data Gap</h2>
          <p>
            Backup generators are everywhere — in hospital basements, airport terminals, fire stations,
            cell towers, data centers, and millions of homes and businesses. They form a massive,
            distributed layer of energy resilience that activates during grid failures. Yet
            <strong>no unified, public database records where these generators are, who operates them,
            or whether they actually function when needed</strong>.
          </p>
          <p>
            This gap has real consequences. Emergency managers cannot assess which neighborhoods will
            retain power during a hurricane. Environmental regulators cannot track diesel generator
            emissions during prolonged outages. Energy equity researchers cannot measure whether
            low-income communities have less backup power access. The information simply does not exist
            at scale.
          </p>

          <h2 id="sec-1-2">1.2 Can Satellites Help?</h2>
          <p>
            We propose an unconventional approach: <strong>using nighttime satellite imagery to detect
            backup generators indirectly</strong>. NASA's Black Marble product (VIIRS VNP46A2) captures
            the brightness of every 500-meter patch of Earth's surface each night. During a power
            outage, most of a city goes dark — but locations with backup generators keep their lights on.
            This brightness anomaly is visible from space.
          </p>
          <p>
            The core idea is simple: if we compare nighttime brightness before and after a disaster,
            areas that stay anomalously bright during a blackout likely have backup power. We formalize
            this as the <em>Resilience Advantage (RA)</em> — the difference in brightness retention
            between areas near known facilities and the surrounding grid-dependent zones.
          </p>
          <p>
            This is an <strong>exploratory project</strong>. We are not claiming to have solved generator
            detection — we are testing how far satellite remote sensing can take us, what it can and
            cannot reveal, and where the fundamental limitations lie.
          </p>

          <h2 id="sec-1-3">1.3 Approach</h2>
          <p>
            We use critical infrastructure locations (hospitals, airports, fire stations) from
            OpenStreetMap as a <strong>weak supervision signal</strong> — these facilities are likely
            to have generators, so nearby pixels serve as positive training labels. The analysis
            proceeds in three stages:
          </p>
          <ul class="detail-list">
            <li><strong>Stage 1 — Interpretive modeling:</strong> Four statistical models (OLS, MixedLM, Logistic, Cox PH) test whether the resilience signal is real and statistically significant.</li>
            <li><strong>Stage 2 — Predictive modeling:</strong> Random Forest + XGBoost models predict pixel-level backup power probability, validated with Leave-One-Event-Out cross-validation across 25 events.</li>
            <li><strong>Stage 3 — Zip-code analysis:</strong> Extending from pixels to policy-relevant geographic units, testing whether facility density correlates with historical outage severity.</li>
          </ul>
          <p>
            The best model (RF + XGBoost ensemble) achieves <strong>LOEO AUC 0.969</strong> with
            spatial features — strong enough to produce useful probability maps. However, an ablation
            removing all facility-location features drops AUC to <strong>0.700</strong>, revealing
            that most predictive power comes from knowing <em>where facilities are</em>, not from
            the satellite signal alone. The pure NTL behavioral signal is real but modest — sufficient
            for exploratory screening at 500m resolution, not yet reliable enough for individual
            building-level detection.
          </p>

          <h2 id="sec-1-4">1.4 Study Areas</h2>
          <p>
            We analyze {{ sortedEvents.length }} disaster events spanning 2016–2023 across 17 U.S. states,
            Puerto Rico, and Turkey. Events were selected based on outage severity (duration > 48 hours,
            > 50K affected), geographic diversity, and satellite data availability.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Location</th><th>Event</th><th>Date</th><th>Affected</th></tr></thead>
              <tbody>
                <tr v-for="ev in sortedEvents" :key="ev.id">
                  <td><span class="dot" :style="{ background: ev.color }" />{{ ev.subtitle }}</td>
                  <td>{{ ev.name }}</td>
                  <td class="mono">{{ ev.date }}</td>
                  <td class="mono">{{ ev.affectedUsers }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h2 id="sec-1-5">1.5 Research Questions</h2>
          <ul class="detail-list">
            <li><strong>Detection:</strong> Can nighttime satellite imagery detect backup generator activation during power outages?</li>
            <li><strong>Signal decomposition:</strong> How much of the predictive signal comes from genuine NTL behavior versus spatial proximity to known facilities?</li>
            <li><strong>Generalization:</strong> Can a model trained on one set of cities predict backup power in unseen cities?</li>
            <li><strong>Limitations:</strong> What spatial resolution, temporal coverage, and data quality constraints bound this approach?</li>
          </ul>

          <div class="callout callout--cyan">
            <span>--</span>
            <div>
              <strong>Advisor:</strong> Prof. Xiaojiang Li (University of Pennsylvania).
              In collaboration with Temple University and Arizona State University. The project
              began with 6 events in Puerto Rico, Florida, and Louisiana, and has expanded to
              25 events across the continental U.S. and Turkey.
            </div>
          </div>

          <div class="takeaway">
            <div class="takeaway__label">KEY TAKEAWAY</div>
            <p class="takeaway__text">
              This project turns a data gap — no generator records — into a detection problem:
              <strong>can we see generators from space?</strong> The answer is a qualified yes.
              Pure nighttime light behavior achieves <strong>AUC 0.700</strong> (above random but
              modest); adding spatial context raises this to <strong>0.969</strong>. The gap between
              these numbers tells us exactly how much the satellite signal contributes versus
              what we already know from facility locations.
            </p>
          </div>
        </template>

        <!-- 02 Data Collection -->
        <template v-if="sectionId === 'data'">

          <div class="data-table" style="margin-bottom:28px">
            <table>
              <thead>
                <tr><th>Dataset</th><th>Role</th><th>Resolution</th><th>Access</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>NASA Black Marble VNP46A2</strong></td>
                  <td>Daily nighttime light imagery</td>
                  <td class="mono">500m, daily</td>
                  <td><a href="https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP46A2" class="inline-link" target="_blank">Google Earth Engine</a></td>
                </tr>
                <tr>
                  <td><strong>EAGLE-I</strong></td>
                  <td>Power outage records (event selection)</td>
                  <td class="mono">County, hourly</td>
                  <td><a href="https://eagle-i.doe.gov" class="inline-link" target="_blank">DOE EAGLE-I</a></td>
                </tr>
                <tr>
                  <td><strong>OpenStreetMap</strong></td>
                  <td>Critical facility locations (hospitals, airports, etc.)</td>
                  <td class="mono">Point / polygon</td>
                  <td><a href="https://overpass-api.de" class="inline-link" target="_blank">Overpass API</a></td>
                </tr>
                <tr>
                  <td><strong>IBTrACS v4</strong></td>
                  <td>Hurricane tracks + wind radii</td>
                  <td class="mono">6-hourly</td>
                  <td><a href="https://www.ncei.noaa.gov/products/international-best-track-archive" class="inline-link" target="_blank">NCEI / NOAA</a></td>
                </tr>
                <tr>
                  <td><strong>Census ACS</strong></td>
                  <td>Population density, median income (Stage 3)</td>
                  <td class="mono">Zip code / tract</td>
                  <td><a href="https://data.census.gov" class="inline-link" target="_blank">Census Bureau</a></td>
                </tr>
                <tr>
                  <td><strong>NLCD</strong></td>
                  <td>Land cover + impervious surface</td>
                  <td class="mono">30m</td>
                  <td><a href="https://www.mrlc.gov/data/nlcd-land-cover-conus-all-years" class="inline-link" target="_blank">MRLC</a></td>
                </tr>
                <tr>
                  <td><strong>ZCTA Boundaries</strong></td>
                  <td>Zip code spatial units (Stage 3)</td>
                  <td class="mono">Polygon</td>
                  <td><a href="https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=ZIP+Code+Tabulation+Areas" class="inline-link" target="_blank">TIGER/Line</a></td>
                </tr>
                <tr>
                  <td><strong>Miami-Dade Permits</strong></td>
                  <td>Generator ground truth validation</td>
                  <td class="mono">Address-level</td>
                  <td><a href="https://gis-mdc.opendata.arcgis.com" class="inline-link" target="_blank">Miami-Dade Open Data</a></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 1: What is Black Marble?                -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-1">2.1 NASA Black Marble — Nighttime Lights from Space</h2>
          <p>
            Our primary data source is <strong>NASA's Black Marble</strong> product suite (VNP46A2),
            which provides daily, gap-filled nighttime light (NTL) imagery at 500-meter resolution.
            The data comes from the VIIRS Day/Night Band sensor aboard the Suomi NPP satellite,
            which orbits Earth ~14 times per day and captures visible-band light emission after sunset.
          </p>
          <p>
            The <strong>daily temporal resolution is critical</strong> for this project. Power outages
            unfold over days to weeks — generators activate within hours of a blackout and may run for
            days before grid power returns. A monthly or annual composite would blur this timeline
            into invisibility. Daily imagery lets us track the exact onset, duration, and spatial
            pattern of brightness anomalies as they evolve, day by day, through the disaster and
            recovery cycle.
          </p>
          <p>
            However, the <strong>500-meter spatial resolution is a significant limitation</strong>.
            A single pixel covers roughly 25 hectares — an area that may contain a hospital,
            surrounding parking lots, residential blocks, and a park. The generator signal from one
            building is mixed with ambient light (or darkness) from everything else in that pixel.
            This means we are detecting aggregate brightness patterns in the neighborhood of a
            facility, not the facility itself. Small generators (residential, single-business) are
            effectively invisible at this resolution; only large institutional generators that
            meaningfully change a 500m pixel's total radiance are detectable.
          </p>
          <p>
            The VNP46A2 product is not raw sensor output — it undergoes extensive processing by
            NASA's Black Marble team: lunar irradiance correction (so moonlight doesn't inflate values),
            atmospheric correction, bidirectional reflectance correction (BRDF), and a cloud gap-filling
            algorithm that interpolates cloud-covered pixels using temporal neighbors. The result is
            a remarkably clean daily snapshot of artificial light on Earth's surface — including the
            lights that stay on during power outages because of backup generators.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Property</th><th>Value</th></tr></thead>
              <tbody>
                <tr><td>Product</td><td class="mono">VNP46A2 (daily) · VNP46A3 (monthly)</td></tr>
                <tr><td>Key layer</td><td class="mono">Gap_Filled_DNB_BRDF-Corrected_NTL</td></tr>
                <tr><td>Resolution</td><td class="mono">500 m / pixel</td></tr>
                <tr><td>Access</td><td class="mono">Google Earth Engine / NASA LAADS DAAC</td></tr>
                <tr><td>Unit</td><td class="mono">nW / cm² / sr</td></tr>
              </tbody>
            </table>
          </div>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 2: EAGLE-I (moved up)                   -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-2">2.2 EAGLE-I Power Outage Records</h2>
          <p>
            The U.S. Department of Energy's EAGLE-I system provides county-level hourly power
            outage counts, which we use for event selection and temporal alignment. EAGLE-I
            aggregates outage reports from utilities across the country, giving us an independent
            ground-truth reference for when and where blackouts occurred. This data source is what
            tells us which disasters caused significant, sustained power outages worth studying
            with satellite imagery.
          </p>
          <p>We screened all severe weather events in EAGLE-I (2014–2023) and ranked them by a
            severity score (peak affected customers × max outage duration). Final selection criteria:</p>
          <ul class="detail-list">
            <li><strong>Outage duration</strong> > 72 hours — sustained enough for daily NTL to capture the outage-recovery arc</li>
            <li><strong>Peak affected</strong> > 100,000 customers — large enough to produce a spatially detectable NTL signal</li>
            <li><strong>County coverage</strong> ≥ 5 counties reporting — ensures the event is not a localized utility failure</li>
            <li><strong>Geographic diversity</strong> — events spanning different U.S. regions (Southeast, Northeast, Midwest, Southern Plains, Pacific Northwest) and disaster types (hurricanes, earthquakes, winter storms, derechos, ice storms)</li>
            <li><strong>Hurricane track verification</strong> — for hurricane events, IBTrACS v4 track data confirms the study area falls within the storm's wind field (R34 radius)</li>
          </ul>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 3: What does it look like? (Maria demo) -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-3">2.3 What Does a Disaster Look Like in NTL?</h2>
          <p>
            Hurricane Maria made landfall in Puerto Rico on September 20, 2017 as a Category 5 storm,
            causing the longest blackout in U.S. history. <strong>Maria is the closest to an ideal
            case study for our method</strong>: the entire island went dark (near-total outage),
            recovery took months (long observation window), and Puerto Rico's tropical latitude
            means relatively low cloud cover compared to mid-latitude events. The NTL signal is
            dramatic and unambiguous — a best-case scenario for satellite detection.
          </p>
          <p>
            The other 14 events present progressively harder challenges: shorter outages (days
            instead of months), partial rather than total blackouts, higher cloud cover in winter
            storms, and urban areas where baseline brightness is so high that generator light
            is a small fraction of the pixel. How well the method transfers from Maria's ideal
            conditions to these harder cases is a key question this project aims to answer.
          </p>
          <p>
            The player below shows daily satellite imagery of the San Juan metropolitan area.
            On the <strong>left</strong>, each frame is a raw NTL image — brighter pixels mean more light.
            On the <strong>right</strong>, each frame shows the <em>change from baseline</em> (delta NTL),
            computed as (day − BAU) / BAU, where BAU is the pre-disaster median composite.
            <strong style="color:var(--red,#ff6b6b)">Red pixels = dimmer than normal</strong> (outage),
            <strong style="color:#6699ff">blue pixels = brighter than normal</strong> (possible generator).
          </p>
          <p>
            Press play and watch: before September 20, the city pulses with normal nighttime activity.
            After the hurricane, the lights vanish almost entirely. Over the following two months,
            brightness returns unevenly — some areas recover quickly (often near hospitals and airports
            with backup power), while others remain dark for weeks.
          </p>

          <div v-if="ntlFrames" class="ntl-player">
            <div class="ntl-player__panels">
              <div class="ntl-player__panel">
                <div class="ntl-player__label">Daily NTL</div>
                <img :src="`${base}data/frames/${ntlFrames.frames[frameIdx].ntl}`" class="ntl-player__img" />
              </div>
              <div class="ntl-player__panel">
                <div class="ntl-player__label">Delta NTL (vs BAU)</div>
                <img :src="`${base}data/frames/${ntlFrames.frames[frameIdx].delta}`" class="ntl-player__img" />
              </div>
            </div>
            <div class="ntl-player__info">
              <span class="ntl-player__date mono">
                {{ ntlFrames.frames[frameIdx].date }}
              </span>
              <span class="ntl-player__phase" :class="ntlFrames.frames[frameIdx].phase">
                {{ ntlFrames.frames[frameIdx].phase === 'pre' ? 'PRE-DISASTER' : 'POST-DISASTER' }}
              </span>
              <span class="mono" style="color:var(--text-muted); font-size:11px">
                Mean NTL: {{ ntlFrames.frames[frameIdx].mean_ntl }} nW/cm²/sr
              </span>
            </div>
            <div class="ntl-player__controls">
              <button class="ntl-player__btn" @click="prevFrame">⏮</button>
              <button class="ntl-player__btn ntl-player__btn--play" @click="togglePlay">
                {{ playing ? '⏸' : '▶' }}
              </button>
              <button class="ntl-player__btn" @click="nextFrame">⏭</button>
              <input type="range" class="ntl-player__slider" min="0" :max="ntlFrames.frames.length - 1" v-model.number="frameIdx" />
              <span class="mono" style="font-size:10px; color:var(--text-dim); min-width:50px; text-align:right">
                {{ frameIdx + 1 }} / {{ ntlFrames.frames.length }}
              </span>
            </div>
            <!-- Colorbar legends -->
            <div class="ntl-player__legends">
              <div class="ntl-player__legend">
                <span style="font-size:10px; color:var(--text-muted)">NTL (nW/cm²/sr)</span>
                <div class="legend-bar legend-bar--hot" />
                <div class="legend-labels"><span>0</span><span>{{ ntlFrames.vmax }}</span></div>
              </div>
              <div class="ntl-player__legend">
                <span style="font-size:10px; color:var(--text-muted)">Delta NTL</span>
                <div class="legend-bar legend-bar--div" />
                <div class="legend-labels"><span>-100%</span><span>0</span><span>+100%</span></div>
              </div>
            </div>
          </div>

          <p>
            This pattern — a sudden NTL collapse followed by gradual, spatially uneven recovery — is
            what we observe across all 25 study events, though the severity and duration vary
            significantly. The bar charts below summarize the daily spatial-mean NTL for each event,
            with <strong style="color:var(--green)">green bars = pre-disaster</strong> and
            <strong style="color:var(--red, #ff6b6b)">red bars = post-disaster</strong>.
          </p>

          <div v-if="cloudStats" class="collapsible" :class="{ expanded: ntlExpanded }">
            <div class="collapsible__content">
              <div class="ntl-charts-grid">
                <div v-for="ev in cloudEvents" :key="ev.id" class="ntl-chart-card" @click="chartModal = { type: 'ntl', ev }" style="cursor:pointer">
                  <div class="ntl-chart-card__header">
                    <span class="dot" :style="{ background: ev.color }" />
                    <strong>{{ ev.subtitle.split(',')[0] }}</strong>
                    <span class="mono" style="color:var(--text-dim); font-size:9px; margin-left:auto">{{ ev.year }}</span>
                  </div>
                  <svg :viewBox="`0 0 ${ntlChartW} ${ntlChartH}`" class="ntl-svg" preserveAspectRatio="xMidYMid meet">
                    <line v-for="y in [0.25, 0.5, 0.75, 1.0]" :key="y"
                      :x1="ntlPad.l" :x2="ntlChartW - ntlPad.r"
                      :y1="ntlY(y, ev.id)" :y2="ntlY(y, ev.id)"
                      stroke="rgba(255,255,255,0.06)" stroke-width="0.5" />
                    <line :x1="ntlX(ev.id, 'split')" :x2="ntlX(ev.id, 'split')"
                      :y1="ntlPad.t" :y2="ntlChartH - ntlPad.b"
                      stroke="rgba(255,100,100,0.6)" stroke-width="1" stroke-dasharray="3 2" />
                    <polyline :points="ntlLinePath(ev.id, 'pre')" fill="none" stroke="rgba(0,229,160,0.8)" stroke-width="1.5" />
                    <polyline :points="ntlLinePath(ev.id, 'post')" fill="none" stroke="rgba(255,107,107,0.8)" stroke-width="1.5" />
                    <polygon :points="ntlAreaPath(ev.id, 'pre')" fill="rgba(0,229,160,0.12)" />
                    <polygon :points="ntlAreaPath(ev.id, 'post')" fill="rgba(255,107,107,0.12)" />
                    <text :x="ntlPad.l + 2" :y="ntlChartH - 2" fill="rgba(0,229,160,0.7)" font-size="7" font-family="monospace">Pre</text>
                    <text :x="ntlX(ev.id, 'split') + 3" :y="ntlChartH - 2" fill="rgba(255,107,107,0.7)" font-size="7" font-family="monospace">Post</text>
                  </svg>
                </div>
              </div>
            </div>
            <div class="collapsible__fade" v-if="!ntlExpanded" />
            <button class="collapsible__toggle" @click="ntlExpanded = !ntlExpanded">
              {{ ntlExpanded ? 'Show Less' : `Show All ${cloudEvents.length} Events` }}
            </button>
          </div>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 4: Cloud coverage                       -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-4">2.4 Cloud Contamination & Quality Control</h2>
          <p>
            A fundamental challenge of optical satellite remote sensing is cloud cover. When clouds
            obscure the ground, the sensor cannot measure surface-emitted light, and the resulting
            pixel values are unreliable. This is especially problematic for disaster studies because
            the very weather systems that cause hurricanes also produce extensive cloud cover in the
            days surrounding landfall — precisely when we most need clear observations.
          </p>
          <p>
            The VNP46A2 product partially addresses this through its gap-filling algorithm, which
            uses temporal interpolation to estimate NTL values for clouded pixels. However, gap-filling
            has limits: when an entire region is cloud-covered for multiple consecutive days, the
            interpolated values become unreliable. We therefore apply a second quality control step
            after download.
          </p>
          <div class="callout callout--cyan">
            <span>--</span>
            <div>
              <strong>Two data strategies:</strong> In the exploratory phase (EDA), we applied
              strict QA-pixel-band masking during GEE export — only retaining genuinely cloud-free
              pixels and excluding cloudy days entirely (threshold: 30% cloud cover). This ensures
              the EDA statistics reflect real observations, not interpolations.
              <br /><br />
              For the predictive modeling phase, we expanded to 25 events using <strong>gap-filled
              imagery</strong> (VNP46A2) to maximize geographic coverage. Gap-filling uses temporal
              interpolation from neighboring days, which may slightly inflate post-disaster brightness
              for days with cloud cover (the algorithm borrows from pre-disaster values). However,
              since our models use <strong>pre/post period averages</strong> rather than single-day
              values, a few interpolated days within a 30–45 day window have minimal impact on the
              aggregate statistics.
            </div>
          </div>

          <!-- Cloud coverage summary table (collapsible) -->
          <div v-if="cloudFrac" class="collapsible" :class="{ expanded: codeExpanded.cloudTable }">
            <div class="collapsible__content">
              <div class="data-table" style="margin:0">
                <table>
                  <thead>
                    <tr>
                      <th>Event</th>
                      <th>Total Days</th>
                      <th>Usable Days</th>
                      <th>Excluded Days</th>
                      <th>Avg Cloud %</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="ev in cloudEvents" :key="ev.id">
                      <td><span class="dot" :style="{ background: ev.color }" />{{ ev.name }}</td>
                      <td class="mono">{{ cloudFrac[ev.dashId]?.summary.total_days ?? '—' }}</td>
                      <td class="mono" style="color:var(--green)">{{ cloudFrac[ev.dashId]?.summary.usable_days ?? '—' }}</td>
                      <td class="mono" :style="{ color: (cloudFrac[ev.dashId]?.summary.excluded_days ?? 0) > 20 ? '#ff6b6b' : 'var(--text)' }">
                        {{ cloudFrac[ev.dashId]?.summary.excluded_days ?? '—' }}
                      </td>
                      <td class="mono" :style="{ color: (cloudFrac[ev.dashId]?.summary.avg_cloud_pct ?? 0) > 40 ? '#ff6b6b' : 'var(--text)' }">
                        {{ cloudFrac[ev.dashId]?.summary.avg_cloud_pct ?? '—' }}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.cloudTable" />
            <button class="collapsible__toggle" @click="codeExpanded.cloudTable = !codeExpanded.cloudTable">
              {{ codeExpanded.cloudTable ? 'Collapse Table' : `Show All ${cloudEvents.length} Events` }}
            </button>
          </div>

          <p>
            The charts below show the daily <strong>cloud fraction</strong> (% of pixels obscured)
            for each event. <span style="color:rgba(0,180,255,0.8)">Blue = usable day</span> (cloud &lt; 30%),
            <span style="color:rgba(255,80,80,0.8)">red = excluded day</span> (cloud &ge; 30%).
            The <span style="color:rgba(255,170,0,0.8)">orange dashed line</span> marks the 30% threshold.
            Note how Michael has the worst cloud contamination (avg 47%), while EQ San Juan
            is the cleanest (avg 20%).
          </p>

          <div v-if="cloudFrac" class="collapsible" :class="{ expanded: cloudExpanded }">
            <div class="collapsible__content">
              <div class="ntl-charts-grid">
                <div v-for="ev in cloudEvents" :key="'cloud-'+ev.id" class="ntl-chart-card" @click="chartModal = { type: 'cloud', ev }" style="cursor:pointer">
                  <div class="ntl-chart-card__header">
                    <span class="dot" :style="{ background: ev.color }" />
                    <strong>{{ ev.subtitle.split(',')[0] }}</strong>
                    <span class="mono" style="color:var(--text-dim); font-size:9px; margin-left:auto">{{ cloudFrac[ev.dashId]?.summary.avg_cloud_pct }}%</span>
                  </div>
                  <svg :viewBox="`0 0 ${ntlChartW} ${ntlChartH}`" class="ntl-svg" preserveAspectRatio="xMidYMid meet">
                    <!-- 30% threshold -->
                    <line :x1="ntlPad.l" :x2="ntlChartW - ntlPad.r"
                      :y1="cloudY(30)" :y2="cloudY(30)"
                      stroke="rgba(255,170,0,0.6)" stroke-width="1" stroke-dasharray="4 2" />
                    <text :x="ntlChartW - ntlPad.r - 2" :y="cloudY(30) - 3" fill="rgba(255,170,0,0.8)" font-size="7" font-family="monospace" text-anchor="end">30% usability threshold</text>
                    <!-- Disaster line -->
                    <line :x1="cfSplitX(ev.dashId)" :x2="cfSplitX(ev.dashId)"
                      :y1="ntlPad.t" :y2="ntlChartH - ntlPad.b"
                      stroke="rgba(255,255,255,0.2)" stroke-width="1" stroke-dasharray="3 2" />
                    <!-- Bars -->
                    <rect v-for="(d, i) in getCfDays(ev.dashId)" :key="'cf'+i"
                      :x="cfBarX(ev.dashId, i)" :y="cloudY(d.cloud_pct)"
                      :width="cfBarW(ev.dashId)" :height="ntlChartH - ntlPad.b - cloudY(d.cloud_pct)"
                      :fill="d.usable ? 'rgba(0,180,255,0.55)' : 'rgba(255,80,80,0.6)'"
                      rx="1" />
                  </svg>
                </div>
              </div>
            </div>
            <div class="collapsible__fade" v-if="!cloudExpanded" />
            <button class="collapsible__toggle" @click="cloudExpanded = !cloudExpanded">
              {{ cloudExpanded ? 'Show Less' : `Show All ${cloudEvents.length} Events` }}
            </button>
          </div>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 5: GEE Download Pipeline                -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-5">2.5 Data Acquisition via Google Earth Engine</h2>
          <p>
            All NTL imagery is downloaded programmatically through the Google Earth Engine (GEE)
            Python API. For each event, we define a bounding box covering the study area, a
            pre-disaster time window (typically 30 days before the event), and a post-disaster
            window (30–90 days after). GEE handles the server-side filtering and clipping, then
            exports each daily image as a GeoTIFF to Google Drive for local processing.
          </p>
          <p>
            This approach is reproducible and scalable — adding a new event requires only
            specifying the bounding box, disaster date, and time windows. The code below shows
            the core download logic for Hurricane Maria:
          </p>
          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.gee }">
            <div class="collapsible__content">
              <div class="code-block">
                <div class="code-block__header"><span class="mono">Python · GEE Download</span></div>
                <pre><code>import ee
ee.Initialize()

# Define study area and time window
bbox = ee.Geometry.Rectangle([west, south, east, north])
pre_start, pre_end = '2017-08-20', '2017-09-19'
post_start, post_end = '2017-09-21', '2017-11-20'

# Load VNP46A2 daily NTL
vnp46 = ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
ntl_band = 'Gap_Filled_DNB_BRDF-Corrected_NTL'

# Filter and download pre-disaster
pre_col = (vnp46
    .filterBounds(bbox)
    .filterDate(pre_start, pre_end)
    .select(ntl_band))

# Export each day as GeoTIFF
for img in pre_col.toList(pre_col.size()).getInfo():
    image = ee.Image(img['id']).select(ntl_band)
    task = ee.batch.Export.image.toDrive(
        image=image.clip(bbox),
        description=f'pre_{img["id"].split("/")[-1]}',
        scale=500,  # 500m resolution
        region=bbox,
        fileFormat='GeoTIFF'
    )
    task.start()</code></pre>
              </div>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.gee" />
            <button class="collapsible__toggle" @click="codeExpanded.gee = !codeExpanded.gee">
              {{ codeExpanded.gee ? 'Collapse Code' : 'Expand Full Code' }}
            </button>
          </div>

          <div class="callout callout--amber">
            <span>⚠️</span>
            <div>
              <strong>Cloud masking strategy:</strong> After download, each daily GeoTIFF is checked:
              pixels with NTL ≤ 0 or NaN are treated as cloud-contaminated/nodata. Days with
              &lt;30% valid pixels are excluded from temporal mean computation. This two-stage
              approach (GEE gap-filling + post-download QC) maximizes usable observations while
              maintaining data quality.
            </div>
          </div>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 7: Generator Permit Data                -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-6">2.6 Generator Permit Records — The Ground Truth Gap</h2>
          <p>
            While no unified generator database exists, many U.S. jurisdictions do require permits
            for generator installation — but these records are fragmented, inconsistent, and hard
            to access. The permit type varies by jurisdiction and generator size:
          </p>
          <ul class="detail-list">
            <li><strong>Air quality permits</strong> — Large diesel generators (typically >50 kW)
              require emissions permits from state environmental agencies (e.g., EPA Title V,
              state-level "minor source" permits). These track fuel type and rated capacity but
              rarely include exact coordinates.</li>
            <li><strong>Building/electrical permits</strong> — Smaller generators require local
              building permits for installation. Miami-Dade County, for example, issues electrical
              permits tagged with "GENERATOR" in the work description. These include precise
              addresses but miss generators installed during original construction.</li>
            <li><strong>Fire department permits</strong> — Some jurisdictions require fire safety
              permits for fuel storage associated with generators, tracked separately from
              building permits.</li>
            <li><strong>Utility interconnection agreements</strong> — Generators capable of
              back-feeding the grid require utility approval, but these records are proprietary.</li>
          </ul>

          <p>Two fundamental challenges prevent assembling a comprehensive generator database:</p>

          <div class="data-table">
            <table>
              <thead><tr><th>Challenge</th><th>Description</th></tr></thead>
              <tbody>
                <tr>
                  <td style="font-weight:600">Historical data loss</td>
                  <td>Many jurisdictions only digitized permit records in the 2000s–2010s.
                    Generators installed in the 1990s or earlier — including large institutional
                    units at hospitals and airports — often have no digital record. The permit
                    captures <em>installation</em>, not <em>existence</em>.</td>
                </tr>
                <tr>
                  <td style="font-weight:600">Inconsistent classification</td>
                  <td>A hospital generator might appear as an "electrical permit," an "air permit,"
                    or a "mechanical permit" depending on the jurisdiction. Residential portable
                    generators may require no permit at all. There is no standard code or category
                    that means "backup generator" across all U.S. counties.</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p>
            Our collaborators at <strong>Arizona State University</strong> and <strong>Temple
            University</strong> have been manually collecting generator permit data from county
            open data portals across our study areas. To date, <strong>Miami-Dade County</strong>
            (Florida) is the only jurisdiction with sufficiently complete, geocoded permit data
            for quantitative validation — yielding 592 generator permits (499 residential,
            93 commercial). This dataset is used in Section 6.7 to validate the predictive model's
            output against real generator locations.
          </p>
          <p>
            Collection efforts for additional counties (Harris County TX, Duval County FL,
            Fulton County GA) are ongoing. The difficulty of assembling this data is itself
            a key motivation for our satellite-based detection approach — if generator locations
            were easy to compile, there would be less need for remote sensing.
          </p>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 6: Generator Permit Data                -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-7">2.7 Critical Infrastructure from OpenStreetMap</h2>
          <p>
            A core challenge of this project is that <strong>no public dataset of backup generator
            locations exists</strong>. We cannot simply look up which buildings have generators — this
            information is proprietary, scattered across utility companies, and rarely georeferenced.
            This is precisely why we turn to satellite imagery in the first place: to <em>infer</em>
            generator presence from the NTL signal rather than relying on direct records.
          </p>
          <p>
            Our proxy approach: we identify <strong>critical infrastructure facilities</strong> —
            hospitals, airports, power plants, fire stations, and police stations — which are
            <em>legally required or highly likely</em> to have backup generators. By analyzing the
            NTL patterns in buffer zones around these facilities, we can detect whether backup
            power is actually activating during outages. The facility locations serve as our
            best available spatial prior for where generators should be.
          </p>
          <p>
            We source facility locations from OpenStreetMap (OSM) via its Overpass API — a free,
            crowd-sourced global geodatabase with generally excellent coverage of critical
            infrastructure in the U.S. and Turkey. For each event, we query all relevant facility
            types within the event's bounding box, extract coordinates and names, and cache the
            results locally. This produces between 50 and 420 facilities per event — from 52 in
            Hatay, Turkey to 420 in Miami, Florida.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Facility Type</th><th>OSM Tag</th><th>Buffer Radius</th><th>Signal Group</th></tr></thead>
              <tbody>
                <tr><td>Hospital</td><td class="mono">amenity=hospital</td><td class="mono">750 m</td><td>High</td></tr>
                <tr><td>Airport</td><td class="mono">aeroway=aerodrome</td><td class="mono">1,250 m</td><td>High</td></tr>
                <tr><td>Power Plant</td><td class="mono">power=plant</td><td class="mono">750 m</td><td>High</td></tr>
                <tr><td>Fire Station</td><td class="mono">amenity=fire_station</td><td class="mono">750 m</td><td>Medium</td></tr>
                <tr><td>Police</td><td class="mono">amenity=police</td><td class="mono">750 m</td><td>Medium</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            Airports receive a larger buffer radius (1,250m vs 750m) because their physical
            footprint spans multiple 500m pixels — runways, terminals, and support buildings
            spread across a much larger area than a single hospital or fire station. The buffer
            must encompass the facility's entire lighting footprint, not just its centroid.
          </p>
          <div class="callout callout--amber">
            <span>--</span>
            <div>
              <strong>Excluded facility types:</strong> Government offices, substations, and
              water treatment plants are queried but excluded from the primary "strict" buffer
              label — they either lack backup generators or don't operate at night.
            </div>
          </div>

          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.overpass }">
            <div class="collapsible__content">
              <div class="code-block">
                <div class="code-block__header"><span class="mono">Python · Overpass API Query</span></div>
                <pre><code>import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tags for each facility type
OSM_QUERIES = {
    "hospital":     'node["amenity"="hospital"]({bbox});way["amenity"="hospital"]({bbox});',
    "aerodrome":    'node["aeroway"="aerodrome"]({bbox});way["aeroway"="aerodrome"]({bbox});',
    "power_plant":  'node["power"="plant"]({bbox});way["power"="plant"]({bbox});',
    "fire_station": 'node["amenity"="fire_station"]({bbox});way["amenity"="fire_station"]({bbox});',
    "police":       'node["amenity"="police"]({bbox});way["amenity"="police"]({bbox});',
}

def fetch_facilities(bbox, ftype, query_template):
    """
    Query Overpass API for a facility type within a bounding box.
    bbox = (south, west, north, east)
    Returns list of {name, lon, lat} dicts.
    """
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    body = query_template.replace("{bbox}", bbox_str)
    ql = f'[out:json][timeout:60];({body});out center tags;'

    resp = requests.post(OVERPASS_URL, data={"data": ql}, timeout=90)
    resp.raise_for_status()

    results = []
    for el in resp.json().get("elements", []):
        if el["type"] == "node":
            lon, lat = el["lon"], el["lat"]
        elif "center" in el:
            lon, lat = el["center"]["lon"], el["center"]["lat"]
        else:
            continue
        name = el.get("tags", {}).get("name", "")
        results.append({"name": name, "lon": lon, "lat": lat})
    return results

# Example: fetch hospitals in San Juan, Puerto Rico
bbox = (18.30, -66.25, 18.52, -65.90)
hospitals = fetch_facilities(bbox, "hospital", OSM_QUERIES["hospital"])
print(f"Found {len(hospitals)} hospitals")
for h in hospitals[:3]:
    print(f"  {h['name'] or 'unnamed'} ({h['lon']:.4f}, {h['lat']:.4f})")</code></pre>
              </div>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.overpass" />
            <button class="collapsible__toggle" @click="codeExpanded.overpass = !codeExpanded.overpass">
              {{ codeExpanded.overpass ? 'Collapse Code' : 'Expand Full Code' }}
            </button>
          </div>

          <!-- ════════════════════════════════════════════════ -->
          <!-- SECTION 8: Cross-Event Heterogeneity            -->
          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-2-8">2.8 Cross-Event Heterogeneity</h2>
          <p>
            Our 15 events span very different contexts — geographies from Caribbean islands to
            the U.S. Midwest, city sizes from Lake Charles (~80K) to Atlanta (~6M metro),
            disaster types (hurricanes, earthquakes, winter storms, derechos), and recovery
            timelines from 2 weeks (Irma) to 11 months (Maria). This heterogeneity is both
            a strength for testing generalizability and a challenge: the "generator signature"
            may manifest differently across these conditions. The LOEO cross-validation in
            Section 6 directly tests whether a universal detection model can work despite
            these differences.
          </p>

          <div class="takeaway">
            <div class="takeaway__label">KEY TAKEAWAY</div>
            <p class="takeaway__text">
              We have no direct data on backup generators — no database tells us which buildings
              have them or whether they activated. Instead, we use satellite-observed nighttime
              light as an indirect signal, and critical facility locations as a spatial prior for
              where generators <em>should</em> be. Combined with 500m resolution, cloud contamination,
              and cross-event diversity, this means individual pixel-level predictions should be
              interpreted with caution. The strength of this approach lies in
              <strong>aggregate statistical patterns</strong> — the consistent tendency for buffer
              zones around critical infrastructure to maintain higher NTL ratios during outages
              across multiple independent disaster events.
            </p>
          </div>
        </template>

        <!-- 03 Pixel Panel -->
        <template v-if="sectionId === 'panel'">
          <p>
            For each event, pre- and post-disaster GeoTIFFs are stacked to compute per-pixel
            mean NTL. Only pixels with <code>pre_mean_ntl > 0.5</code> are retained.
            Buffer zones are rasterized in UTM coordinates at 750 m (hospitals, fire stations,
            power plants) and 1,250 m (airports). A cKDTree assigns each pixel its nearest
            facility type and distance.
          </p>
          <div class="formula-block">
            <div class="formula">delta_ntl = (post_mean − pre_mean) / pre_mean</div>
            <div class="formula__caption">Relative NTL change; negative values indicate outage/damage</div>
          </div>

          <h3>Panel Schema</h3>
          <div class="data-table">
            <table>
              <thead><tr><th>Column</th><th>Description</th></tr></thead>
              <tbody>
                <tr><td class="mono">event_id</td><td>Event identifier (e.g. Maria_SanJuan)</td></tr>
                <tr><td class="mono">pre_mean_ntl</td><td>Mean NTL over pre-disaster window</td></tr>
                <tr><td class="mono">post_mean_ntl</td><td>Mean NTL over post-disaster window</td></tr>
                <tr><td class="mono">delta_ntl</td><td>Relative NTL change</td></tr>
                <tr><td class="mono">in_buffer_strict</td><td>1 if inside high/medium-signal buffer</td></tr>
                <tr><td class="mono">nearest_fac_type</td><td>Nearest facility type</td></tr>
                <tr><td class="mono">dist_to_facility</td><td>Distance to nearest facility (metres)</td></tr>
                <tr><td class="mono">city_pre_mean</td><td>City-level mean NTL (floor-effect control)</td></tr>
              </tbody>
            </table>
          </div>

          <h3>Construction Pipeline</h3>
          <ul class="detail-list">
            <li>Download pre/post daily GeoTIFFs via Google Earth Engine</li>
            <li>Stack and compute per-pixel temporal mean (excluding cloud-masked days)</li>
            <li>Filter: retain only pixels with pre_mean_ntl > 0.5 nW/cm²/sr</li>
            <li>Fetch facility POIs from OpenStreetMap Overpass API</li>
            <li>Reproject to local UTM, create buffer polygons, rasterize to pixel grid</li>
            <li>Build cKDTree for nearest-facility assignment (type + distance)</li>
            <li>Merge into a single parquet panel: 15,539 rows × 9 events</li>
          </ul>
        </template>

        <!-- 04 Exploratory Data Analysis -->
        <template v-if="sectionId === 'eda'">

          <h2 id="sec-4-1">3.1 Key Definitions</h2>
          <p>
            Before analyzing the data, we need to define the metrics that quantify how well
            an area maintains nighttime brightness during a power outage. These definitions
            are central to everything that follows.
          </p>

          <h3>Business-As-Usual (BAU) Baseline</h3>
          <p>
            For each pixel, the <strong>BAU</strong> (Business-As-Usual) is the median NTL
            brightness across all pre-disaster observation days. This represents "normal"
            nighttime brightness — what the pixel would look like if no disaster had occurred.
            We use the median rather than the mean to reduce sensitivity to individual cloudy
            or anomalous days.
          </p>
          <div class="formula-block">
            <div class="formula">BAU = median(NTL<sub>pre,day1</sub>, NTL<sub>pre,day2</sub>, ..., NTL<sub>pre,dayN</sub>)</div>
            <div class="formula__caption">Per-pixel temporal median over the pre-disaster window</div>
          </div>

          <h3>Resilience Ratio (R)</h3>
          <p>
            The <strong>Resilience Ratio</strong> measures how much of a pixel's normal brightness
            is maintained on a given post-disaster day. It is defined as:
          </p>
          <div class="formula-block">
            <div class="formula">R = NTL<sub>post,day</sub> / BAU</div>
            <div class="formula__caption">R = 1.0 means full brightness maintained; R = 0 means complete darkness</div>
          </div>
          <p>
            An R value of <strong>1.0</strong> means the pixel is as bright as its pre-disaster
            baseline — either the power grid never failed, or backup generators are sustaining
            the lights. An R of <strong>0.3</strong> means only 30% of normal brightness remains.
            Values above 1.0 are possible (e.g., emergency lighting, reconstruction activity)
            but uncommon.
          </p>

          <h3>Delta NTL</h3>
          <p>
            For the pixel panel, we also compute a summary metric across the entire post-disaster
            window:
          </p>
          <div class="formula-block">
            <div class="formula">delta_ntl = (post_mean − pre_mean) / pre_mean</div>
            <div class="formula__caption">Negative values indicate NTL loss; -0.5 means a 50% drop in brightness</div>
          </div>

          <h3>Damage Label</h3>
          <p>
            For classification, pixels are labeled as <strong>"damaged"</strong> if their
            <code>delta_ntl &lt; -0.10</code> (more than 10% brightness loss). This threshold
            was chosen to exclude minor fluctuations while capturing meaningful outage impacts.
          </p>

          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-4-2">3.2 Buffer Zone vs. Non-Buffer Comparison</h2>
          <p>
            The core analysis compares NTL recovery patterns between two groups of pixels:
          </p>
          <ul class="detail-list">
            <li><strong>Buffer pixels</strong> — within 750m (hospitals, fire stations, power plants)
              or 1250m (airports) of a critical facility</li>
            <li><strong>Non-buffer pixels</strong> — all other urban pixels in the study area,
              not near any known critical infrastructure</li>
          </ul>
          <p>
            If backup generators are producing a detectable NTL signal, we expect buffer pixels
            to have <strong>higher R values</strong> than non-buffer pixels during the post-disaster
            period. This difference is the <em>Resilience Advantage (RA)</em>:
          </p>
          <div class="formula-block">
            <div class="formula">RA = R&#772;<sub>buffer</sub> − R&#772;<sub>non-buffer</sub></div>
            <div class="formula__caption">Positive RA = buffer zones recover faster or maintain brightness better</div>
          </div>
          <p>
            The <RouterLink to="/charts" class="inline-link">Recovery Charts</RouterLink> page
            visualizes these daily R curves for all 9 events. In most events, the buffer curve
            (green) consistently sits above the non-buffer curve (blue) during the post-disaster
            window — but the magnitude varies considerably.
          </p>

          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-4-3">3.3 The Floor Effect</h2>
          <p>
            The <strong>floor effect</strong> is the most important confound in this analysis,
            and understanding it is essential to interpreting results correctly.
          </p>
          <p>
            In large cities like Miami and San Juan, critical infrastructure (hospitals, airports)
            tends to be located in brighter commercial/urban districts. Buffer pixels start with
            high BAU values, and even a large percentage drop still leaves them brighter than
            surrounding residential areas. The buffer advantage is easy to detect.
          </p>
          <p>
            But in <strong>smaller cities</strong> like Lake Charles and Panama City, the situation
            reverses: hospitals and fire stations are often in the <em>darker</em> parts of
            the urban core. Their BAU values are low to begin with. When the grid fails, these
            pixels can drop to near-zero NTL — creating a "floor" below which they cannot fall
            further. This means <strong>buffer pixels in small cities may actually show LOWER
            R values than non-buffer pixels</strong>, not because they lack generators, but because
            their baseline was already dim.
          </p>
          <div class="callout callout--amber">
            <span>⚠️</span>
            <div>
              <strong>Why this matters:</strong> A naive comparison of R<sub>buffer</sub> vs
              R<sub>non-buffer</sub> in small cities would conclude that critical infrastructure
              has <em>worse</em> resilience — the opposite of reality. The floor effect creates
              a systematic bias that must be controlled for in any statistical or predictive model.
            </div>
          </div>
          <!-- Floor Effect Visualization -->
          <h3>Floor Effect — Empirical Evidence</h3>
          <p>
            The table and chart below show this effect clearly with real data from our pixel panel.
            In <strong>large cities</strong>, buffer pixels start brighter (6.69 vs 4.18 nW/cm²/sr)
            and show slightly less NTL loss. But in <strong>small cities</strong>, buffer pixels
            actually show <em>more</em> loss than non-buffer — the floor effect in action.
          </p>

          <div v-if="edaStats" class="data-table" style="margin:16px 0">
            <table>
              <thead><tr><th>City Size</th><th>Cities</th><th>Zone</th><th>Mean Pre-NTL</th><th>Mean Delta NTL</th><th>N Pixels</th></tr></thead>
              <tbody>
                <tr v-for="d in edaStats.floor" :key="d.city_size+d.in_buffer">
                  <td><strong>{{ d.city_size }}</strong></td>
                  <td style="font-size:11px; color:var(--text-muted)">{{ floorCities[d.city_size] }}</td>
                  <td>{{ d.in_buffer ? 'Buffer' : 'Non-Buffer' }}</td>
                  <td class="mono">{{ d.mean_pre_ntl }} nW/cm²/sr</td>
                  <td class="mono" :style="{ color: d.mean_delta < -0.1 ? '#ff6b6b' : d.mean_delta > 0 ? 'var(--green)' : 'var(--text)' }">
                    {{ (d.mean_delta * 100).toFixed(1) }}%
                  </td>
                  <td class="mono">{{ d.n.toLocaleString() }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Floor Effect Bar Chart -->
          <div v-if="edaStats" class="eda-chart-card">
            <div class="ntl-chart-card__header">
              <strong>Baseline Brightness: Buffer vs Non-Buffer by City Size</strong>
            </div>
            <svg viewBox="0 0 520 220" class="ntl-svg" preserveAspectRatio="xMidYMid meet">
              <!-- Y axis label -->
              <text x="10" y="100" font-size="9" fill="var(--text-dim)" font-family="monospace" transform="rotate(-90, 10, 100)" text-anchor="middle">Pre-NTL (nW/cm²/sr)</text>
              <!-- Grid lines -->
              <line v-for="v in [0,2,4,6,8]" :key="v" x1="40" x2="500" :y1="180 - v*20" :y2="180 - v*20" stroke="rgba(255,255,255,0.06)" stroke-width="0.5"/>
              <text v-for="v in [0,2,4,6,8]" :key="'l'+v" x="36" :y="184 - v*20" text-anchor="end" font-size="9" font-family="monospace" fill="var(--text-dim)">{{v}}</text>

              <!-- Bars for each city size -->
              <template v-for="(cs, ci) in floorGroups" :key="cs.key">
                <!-- Group label -->
                <text :x="100 + ci * 160" y="205" text-anchor="middle" font-size="12" fill="var(--text-bright)" font-weight="600">{{ cs.label }}</text>
                <text :x="100 + ci * 160" y="217" text-anchor="middle" font-size="8" fill="var(--text-dim)">{{ cs.cities }}</text>
                <!-- Non-buffer bar -->
                <rect :x="65 + ci*160" :width="30" rx="2"
                  :y="180 - floorVal(cs.key, false) * 20"
                  :height="floorVal(cs.key, false) * 20"
                  fill="rgba(0,212,255,0.5)"/>
                <!-- Buffer bar -->
                <rect :x="100 + ci*160" :width="30" rx="2"
                  :y="180 - floorVal(cs.key, true) * 20"
                  :height="floorVal(cs.key, true) * 20"
                  fill="rgba(0,229,160,0.6)"/>
                <!-- Value labels -->
                <text :x="80 + ci*160" :y="176 - floorVal(cs.key, false) * 20" text-anchor="middle" font-size="9" font-family="monospace" fill="var(--cyan)">{{ floorVal(cs.key, false).toFixed(1) }}</text>
                <text :x="115 + ci*160" :y="176 - floorVal(cs.key, true) * 20" text-anchor="middle" font-size="9" font-family="monospace" fill="var(--green)">{{ floorVal(cs.key, true).toFixed(1) }}</text>
                <!-- Ratio label -->
                <text :x="100 + ci*160" :y="12" text-anchor="middle" font-size="9" font-family="monospace" fill="var(--text-muted)">
                  {{ (floorVal(cs.key, true) / Math.max(floorVal(cs.key, false), 0.01)).toFixed(1) }}×
                </text>
              </template>

              <!-- Legend -->
              <rect x="380" y="2" width="12" height="8" fill="rgba(0,212,255,0.5)" rx="1"/>
              <text x="396" y="9" font-size="9" fill="var(--text-muted)">Non-Buffer</text>
              <rect x="450" y="2" width="12" height="8" fill="rgba(0,229,160,0.6)" rx="1"/>
              <text x="466" y="9" font-size="9" fill="var(--text-muted)">Buffer</text>
            </svg>
            <p style="font-size:12px; color:var(--text-muted); margin:8px 0 0; line-height:1.5">
              In large cities, buffer zones are <strong style="color:var(--green)">1.6× brighter</strong> than non-buffer.
              In small cities, the ratio drops to 1.9× — but absolute brightness is much lower (2.4 vs 4.2), compressing the available signal range.
            </p>
          </div>

          <!-- Per-event RA chart -->
          <h3>Resilience Advantage by Event</h3>
          <p>
            The chart below shows the raw Resilience Advantage (RA = delta_buffer - delta_non-buffer)
            for each event. Positive values (green) mean buffer zones lost less brightness;
            negative values (red) mean buffer zones lost <em>more</em> — the floor effect.
          </p>
          <div v-if="edaStats" class="eda-chart-card">
            <svg viewBox="0 0 500 220" class="ntl-svg" preserveAspectRatio="xMidYMid meet">
              <!-- Center line (RA=0) -->
              <line x1="40" x2="490" y1="110" y2="110" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
              <text x="35" y="113" text-anchor="end" font-size="8" font-family="monospace" fill="var(--text-dim)">0</text>
              <!-- Grid -->
              <line x1="40" x2="490" y1="60" y2="60" stroke="rgba(255,255,255,0.05)" stroke-width="0.5"/>
              <text x="35" y="63" text-anchor="end" font-size="8" font-family="monospace" fill="var(--text-dim)">+5%</text>
              <line x1="40" x2="490" y1="160" y2="160" stroke="rgba(255,255,255,0.05)" stroke-width="0.5"/>
              <text x="35" y="163" text-anchor="end" font-size="8" font-family="monospace" fill="var(--text-dim)">-5%</text>

              <!-- Bars -->
              <template v-for="(ev, i) in sortedEdaEvents" :key="ev.event_id">
                <rect
                  :x="55 + i * 48" :width="32" rx="2"
                  :y="ev.ra >= 0 ? 110 - ev.ra * 1000 : 110"
                  :height="Math.abs(ev.ra) * 1000"
                  :fill="ev.ra >= 0 ? 'rgba(0,229,160,0.6)' : 'rgba(255,107,107,0.6)'"
                />
                <!-- Value -->
                <text :x="71 + i * 48" :y="ev.ra >= 0 ? 106 - ev.ra * 1000 : 124 + Math.abs(ev.ra) * 1000"
                  text-anchor="middle" font-size="8" font-family="monospace"
                  :fill="ev.ra >= 0 ? 'var(--green)' : '#ff6b6b'">
                  {{ (ev.ra * 100).toFixed(1) }}%
                </text>
                <!-- Label -->
                <text :x="71 + i * 48" y="200" text-anchor="middle" font-size="8" fill="var(--text-muted)"
                  transform-origin="center" :transform="`rotate(-30, ${71 + i * 48}, 200)`">
                  {{ ev.event_id.replace('_',' ').replace('Earthquake','EQ') }}
                </text>
                <!-- City size dot -->
                <circle :cx="71 + i * 48" y="210" :cy="212"  r="3"
                  :fill="ev.city_size === 'large' ? 'var(--cyan)' : ev.city_size === 'medium' ? 'var(--green)' : '#ffaa00'" />
              </template>

              <!-- Legend -->
              <circle cx="350" cy="10" r="3" fill="var(--cyan)"/><text x="356" y="13" font-size="8" fill="var(--text-muted)">Large</text>
              <circle cx="390" cy="10" r="3" fill="var(--green)"/><text x="396" y="13" font-size="8" fill="var(--text-muted)">Medium</text>
              <circle cx="440" cy="10" r="3" fill="#ffaa00"/><text x="446" y="13" font-size="8" fill="var(--text-muted)">Small</text>
            </svg>
          </div>

          <p>
            We address the floor effect through several mechanisms:
          </p>
          <ul class="detail-list">
            <li><strong>City-level normalization</strong> — <code>ntl_relative = pre_mean_ntl / city_pre_mean</code> measures brightness relative to the city average</li>
            <li><strong>Below-median indicator</strong> — <code>below_city_median</code> flags pixels in the darker half of their city</li>
            <li><strong>Interaction terms</strong> — <code>below_median_x_group</code> allows the model to learn different patterns for dark-zone vs. bright-zone pixels by facility type</li>
            <li><strong>City size code</strong> — <code>city_size_code</code> (large=0, medium=1, small=2) as a direct control variable</li>
          </ul>

          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-4-4">3.4 Facility Type Differences</h2>
          <p>
            Not all critical facilities produce equal NTL signals. Our analysis groups facilities
            into three tiers based on expected backup power capacity:
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Group</th><th>Facility Types</th><th>Expected Signal</th><th>Rationale</th></tr></thead>
              <tbody>
                <tr>
                  <td><strong>Group 1 — High Signal</strong></td>
                  <td class="mono">hospital, aerodrome, power_plant</td>
                  <td style="color:var(--green)">Strong</td>
                  <td>Large facilities with industrial-scale generators; legally required backup power; 24/7 operations produce consistent nighttime light</td>
                </tr>
                <tr>
                  <td><strong>Group 2 — Medium Signal</strong></td>
                  <td class="mono">fire_station, police</td>
                  <td style="color:var(--cyan)">Moderate</td>
                  <td>Smaller facilities with portable or partial generators; may not illuminate enough area to significantly change a 500m pixel</td>
                </tr>
                <tr>
                  <td><strong>Group 3 — Excluded</strong></td>
                  <td class="mono">government, substation, water_works</td>
                  <td style="color:var(--text-muted)">Weak / None</td>
                  <td>No consistent nighttime operations; substations don't generate light; used only as sensitivity test baseline</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            The EDA confirms this hierarchy: model-predicted probabilities show clear stratification
            by facility group, with Group 1 buffer pixels consistently scoring highest.
            Interestingly, the <em>individual</em> facility type indicators for Group 1
            (near_hospital, near_aerodrome, near_power_plant) show zero feature importance in
            the predictive model — because they are perfectly correlated with the buffer label
            itself. The meaningful variation comes from <strong>Group 2 vs. Group 1</strong>
            and the <strong>fac_group ordinal</strong> feature.
          </p>

          <!-- Facility Type Data Table -->
          <h3>Facility Type Statistics</h3>
          <div v-if="edaStats" class="data-table" style="margin:16px 0">
            <table>
              <thead><tr><th>Facility Type</th><th>Total Pixels</th><th>Buffer Pixels</th><th>Mean Pre-NTL</th><th>Mean Delta (All)</th><th>Mean Delta (Buffer)</th></tr></thead>
              <tbody>
                <tr v-for="d in edaStats.facilities" :key="d.type">
                  <td><strong>{{ d.type }}</strong></td>
                  <td class="mono">{{ d.n_total.toLocaleString() }}</td>
                  <td class="mono">{{ d.n_buffer.toLocaleString() }}</td>
                  <td class="mono">{{ d.mean_pre_ntl }}</td>
                  <td class="mono" :style="{ color: d.mean_delta < -0.1 ? '#ff6b6b' : 'var(--text)' }">
                    {{ (d.mean_delta * 100).toFixed(1) }}%
                  </td>
                  <td class="mono" :style="{ color: d.buf_delta && d.buf_delta < -0.1 ? '#ff6b6b' : 'var(--text)' }">
                    {{ d.buf_delta != null ? (d.buf_delta * 100).toFixed(1) + '%' : '—' }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-4-5">3.5 City Size & Cross-Event Variation</h2>
          <p>
            The 9 events span three city size categories, each with distinct characteristics
            that affect the NTL signal:
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>City Size</th><th>Events</th><th>Characteristics</th></tr></thead>
              <tbody>
                <tr>
                  <td><strong>Large</strong></td>
                  <td>San Juan (Maria, EQ), Miami (Irma), New Orleans (Ida)</td>
                  <td>Bright urban cores; floor effect minimal; strong baseline NTL contrast between buffer and non-buffer</td>
                </tr>
                <tr>
                  <td><strong>Medium</strong></td>
                  <td>Fort Myers (Ian), Hatay (EQ)</td>
                  <td>Mixed brightness; some floor effect in downtown areas; moderate signal</td>
                </tr>
                <tr>
                  <td><strong>Small</strong></td>
                  <td>Lake Charles (Laura), Panama City (Michael), Charlotte Harbor (Ian)</td>
                  <td>Low baseline NTL; strong floor effect; infrastructure in dark zones; hardest events for the model</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            Additionally, disaster type introduces variation: <strong>hurricanes</strong> produce
            widespread, sustained outages with gradual recovery, while <strong>earthquakes</strong>
            cause more sudden, spatially concentrated damage. The two earthquake events
            (San Juan 2020, Hatay 2023) are geographically coupled only to their specific
            regions, making cross-event earthquake generalization particularly challenging.
          </p>

          <!-- ════════════════════════════════════════════════ -->
          <h2 id="sec-4-6">3.6 Key EDA Findings</h2>

          <div class="takeaway">
            <div class="takeaway__label">KEY FINDINGS</div>
            <p class="takeaway__text">
              <strong>1. The resilience signal is real but subtle.</strong> Buffer zones show
              consistently higher R values in most events, but the magnitude (RA ≈ 5–20%) is
              small relative to pixel-level noise.<br /><br />
              <strong>2. Floor effect is the dominant confound.</strong> In small cities, raw
              buffer-vs-non-buffer comparisons can be misleading or reversed. Any valid analysis
              must control for baseline brightness.<br /><br />
              <strong>3. Facility type matters.</strong> Hospitals and airports (Group 1) show the
              strongest resilience signal. Fire stations and police (Group 2) contribute a weaker
              but detectable signal. Government buildings and substations (Group 3) show no
              meaningful generator signature.<br /><br />
              <strong>4. City size systematically affects model performance.</strong> Large-city
              events are easiest to predict; small-city events with strong floor effects are hardest.
              This has direct implications for cross-event generalization.
            </p>
          </div>
        </template>

        <!-- 05 Interpretive Modeling -->
        <template v-if="sectionId === 'interpretive'">

          <!-- ═══ 5.0 Why Interpretive Modeling ═══ -->
          <h2 id="sec-5-1">4.1 Why Interpretive Modeling First?</h2>
          <p>
            Before building a predictive model, we need to answer a more fundamental question:
            <strong>is there actually a detectable signal?</strong> The EDA shows that buffer
            zones near facilities have higher resilience ratios on average — but is that
            difference statistically significant, or could it be explained by confounding
            variables like urban brightness and land use?
          </p>
          <p>
            Interpretive modeling serves three purposes in our pipeline:
          </p>
          <ul class="detail-list">
            <li><strong>Validate the signal exists</strong> — If four different statistical
              models all find a significant effect in the same direction, the signal is unlikely
              to be an artifact of any single model's assumptions.</li>
            <li><strong>Understand what drives it</strong> — OLS interaction terms reveal that
              the effect depends on baseline brightness (floor effect). MixedLM shows the
              effect survives event-level clustering. Logistic regression gives intuitive
              odds ratios. Cox PH adds the time dimension.</li>
            <li><strong>Guide feature engineering</strong> — The confounds and interactions
              discovered here directly inform which features we engineer for the predictive
              models in Stage 2. The floor effect motivates city-level normalization; the
              land-use confound motivates NLCD controls; the facility-type variation
              motivates group-level features.</li>
          </ul>
          <p>
            This is the <strong>triangulation</strong> approach: four models, same hypothesis,
            different angles. Only when all four point in the same direction can we confidently
            proceed to prediction.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Model</th><th>DV</th><th>Question</th><th>Unique Contribution</th></tr></thead>
              <tbody>
                <tr><td><strong>OLS</strong></td><td class="mono">delta_ntl</td><td>How much less NTL decline?</td><td>Baseline effect size; reveals interaction with brightness</td></tr>
                <tr><td><strong>MixedLM</strong></td><td class="mono">delta_ntl + u<sub>j</sub></td><td>Same, with clustering correction</td><td>Confirms OLS isn't inflated by within-event correlation</td></tr>
                <tr><td><strong>Logistic</strong></td><td class="mono">is_damaged (binary)</td><td>Lower damage probability?</td><td>Intuitive OR; AUC for discrimination; LOEO direction test</td></tr>
                <tr><td><strong>Cox PH</strong></td><td class="mono">recovery_days</td><td>Faster recovery?</td><td>Time dimension; independent from cross-section models</td></tr>
              </tbody>
            </table>
          </div>

          <!-- ═══ 5.2 Specification ═══ -->
          <h2 id="sec-5-2">4.2 Specification & Controls</h2>
          <p>
            All models share a core specification (n = 10,306 pixels, 6 events) with two variants
            to test the impact of land-use confounding:
          </p>
          <div class="formula-block">
            <div class="formula">Y<sub>i</sub> = &beta;<sub>0</sub> + &beta;<sub>1</sub> &middot; in_buffer + &beta;<sub>2</sub> &middot; pre_mean_ntl + &beta;<sub>3</sub> &middot; (in_buffer &times; pre_mean_ntl) + &gamma; &middot; C(event_id) [+ &delta; &middot; NLCD] + &epsilon;</div>
            <div class="formula__caption">The interaction term &beta;<sub>3</sub> captures the floor effect: generator signal strength depends on baseline brightness</div>
          </div>
          <ul class="detail-list">
            <li><strong>no_nlcd</strong> — baseline: pre_mean_ntl, event fixed effects, interaction term</li>
            <li><strong>with_nlcd</strong> — adds NLCD land-use dummies (developed 22/23/24), OSM facility density, cloud quality proxy</li>
          </ul>

          <!-- ═══ 5.3 Model Details (collapsible) ═══ -->
          <h2 id="sec-5-3">4.3 OLS — Baseline Effect Size</h2>
          <div class="formula-block">
            <div class="formula">delta_ntl<sub>i</sub> = &beta;<sub>0</sub> + &beta;<sub>1</sub> &middot; in_buffer<sub>i</sub> + &beta;<sub>2</sub> &middot; pre_mean_ntl<sub>i</sub> + &beta;<sub>3</sub> &middot; (in_buffer<sub>i</sub> &times; pre_mean_ntl<sub>i</sub>) + &gamma; &middot; NLCD<sub>i</sub> + &epsilon;<sub>i</sub></div>
            <div class="formula__caption">HC1 robust standard errors; &beta;<sub>1</sub> = average buffer effect; &beta;<sub>3</sub> = floor effect interaction</div>
          </div>
          <p>
            OLS establishes the baseline: buffer pixels show <strong>+2.8% less NTL decline</strong>
            (p = 0.070, marginal). The interaction term <code>in_buffer &times; pre_mean_ntl</code>
            is the key discovery — with NLCD controls, it becomes highly significant (p = 0.0002),
            revealing that the <strong>generator effect is stronger in brighter areas</strong> and
            invisible in dim small cities. This is the statistical fingerprint of the floor effect.
          </p>
          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.ols }">
            <div class="collapsible__content">
              <div class="data-table">
                <table>
                  <thead><tr><th>Variable</th><th>no_nlcd coef</th><th>p</th><th>with_nlcd coef</th><th>p</th></tr></thead>
                  <tbody>
                    <tr><td class="mono">in_buffer</td><td class="mono" style="color:var(--green)">+0.028</td><td class="mono">0.070</td><td class="mono" style="color:#ff6b6b">-0.024</td><td class="mono">0.122</td></tr>
                    <tr><td class="mono">in_buffer &times; pre_mean_ntl</td><td class="mono">~0</td><td class="mono">n.s.</td><td class="mono" style="color:var(--green)">+0.010</td><td class="mono" style="color:var(--green); font-weight:700">0.0002</td></tr>
                    <tr><td class="mono">C(event_id)[maria]</td><td class="mono" style="color:#ff6b6b">-0.50</td><td class="mono">&lt;0.001</td><td colspan="2">Island grid collapse</td></tr>
                    <tr><td class="mono">C(event_id)[michael]</td><td class="mono" style="color:#ff6b6b">-0.11</td><td class="mono">&lt;0.001</td><td colspan="2">Small city, Cat 5</td></tr>
                  </tbody>
                </table>
              </div>
              <p>
                Event fixed effects match physical expectations: Maria is the most extreme (-0.50),
                large-city events (Irma, Ida) show less loss. The model's R² is low — expected for
                an explanatory model, not a pixel-level predictor.
              </p>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.ols" />
            <button class="collapsible__toggle" @click="codeExpanded.ols = !codeExpanded.ols">
              {{ codeExpanded.ols ? 'Collapse Details' : 'Show Full OLS Results' }}
            </button>
          </div>

          <h2 id="sec-5-4">4.4 MixedLM — Clustering Correction</h2>
          <div class="formula-block">
            <div class="formula">delta_ntl<sub>ij</sub> = &beta;<sub>0</sub> + &beta;<sub>1</sub> &middot; in_buffer<sub>ij</sub> + &beta;<sub>2</sub> &middot; pre_mean_ntl<sub>ij</sub> + &gamma; &middot; NLCD<sub>ij</sub> + u<sub>j</sub> + &epsilon;<sub>ij</sub></div>
            <div class="formula__caption">i = pixel, j = event; u<sub>j</sub> ~ N(0, &sigma;²<sub>u</sub>) event-level random intercept</div>
          </div>
          <p>
            Pixels within the same disaster event share the same grid, weather, and recovery
            trajectory — they are not independent. MixedLM adds event-level random intercepts
            to correct for this. Result: <strong>same coefficients as OLS, but p-value improves
            from 0.070 to 0.020</strong> — the buffer effect is genuine, not inflated by
            pseudo-replication.
          </p>
          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.mixed }">
            <div class="collapsible__content">
              <div class="data-table">
                <table>
                  <thead><tr><th>Metric</th><th>no_nlcd</th><th>with_nlcd</th></tr></thead>
                  <tbody>
                    <tr><td>in_buffer coef</td><td class="mono" style="color:var(--green)">+0.028 (p=0.020)</td><td class="mono" style="color:#ff6b6b">-0.024 (p=0.045)</td></tr>
                    <tr><td>Random intercept var (&sigma;²<sub>u</sub>)</td><td class="mono" colspan="2">&asymp; 0 (singular)</td></tr>
                  </tbody>
                </table>
              </div>
              <p>
                The random intercept variance is effectively zero — event-level differences are
                fully captured by the fixed effects. The model degenerates to "OLS with clustered
                standard errors," which is informative: it tells us the six events don't have
                residual systematic differences beyond what <code>in_buffer</code> and
                <code>pre_mean_ntl</code> already explain.
              </p>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.mixed" />
            <button class="collapsible__toggle" @click="codeExpanded.mixed = !codeExpanded.mixed">
              {{ codeExpanded.mixed ? 'Collapse Details' : 'Show Full MixedLM Results' }}
            </button>
          </div>

          <h2 id="sec-5-5">4.5 Logistic Regression — Damage Probability</h2>
          <div class="formula-block">
            <div class="formula">log[P(damaged<sub>i</sub>=1) / (1 - P(damaged<sub>i</sub>=1))] = &beta;<sub>0</sub> + &beta;<sub>1</sub> &middot; in_buffer<sub>i</sub> + &beta;<sub>2</sub> &middot; pre_mean_ntl<sub>i</sub> + &gamma; &middot; NLCD<sub>i</sub></div>
            <div class="formula__caption">damaged = 1 if delta_ntl &lt; -10%; OR = exp(&beta;<sub>1</sub>); OR &lt; 1 = protective effect</div>
          </div>
          <p>
            Logit converts the question to binary: "was this pixel damaged (&gt;10% NTL loss)?"
            Buffer pixels have <strong>OR = 0.68 (p &lt; 0.001)</strong> — 32% lower odds of damage.
            The AUC of 0.72 means the model correctly ranks a random damaged/undamaged pixel pair
            72% of the time. Critically, in LOEO cross-validation, the <strong>direction is correct
            in all 6 folds</strong> (100% sign consistency), even though AUC drops to 0.455.
          </p>
          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.logit }">
            <div class="collapsible__content">
              <div class="data-table">
                <table>
                  <thead><tr><th>Metric</th><th>no_nlcd</th><th>with_nlcd</th></tr></thead>
                  <tbody>
                    <tr><td>OR (in_buffer)</td><td class="mono" style="color:var(--green)">0.683 (p&lt;0.001)</td><td class="mono">1.178 (p=0.105)</td></tr>
                    <tr><td>AUC (sample)</td><td class="mono">0.719</td><td class="mono">0.749</td></tr>
                    <tr><td>LOEO AUC</td><td class="mono" style="color:#ff6b6b" colspan="2">0.455 (near random)</td></tr>
                    <tr><td>LOEO sign consistency</td><td class="mono" style="color:var(--green)" colspan="2">6/6 = 100%</td></tr>
                    <tr><td>Robustness (-5% to -20%)</td><td class="mono" colspan="2">OR range: 0.60–0.72, direction never changes</td></tr>
                  </tbody>
                </table>
              </div>
              <p>
                The 100% sign consistency in LOEO is a crucial finding: even when the model can't
                accurately <em>quantify</em> the effect for a new event, it always gets the
                <em>direction</em> right. This validates <code>in_buffer</code> as a meaningful
                weak-supervision label for Phase 3 predictive modeling.
              </p>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.logit" />
            <button class="collapsible__toggle" @click="codeExpanded.logit = !codeExpanded.logit">
              {{ codeExpanded.logit ? 'Collapse Details' : 'Show Full Logistic Results' }}
            </button>
          </div>

          <h2 id="sec-5-6">4.6 Cox PH — Recovery Speed</h2>
          <div class="formula-block">
            <div class="formula">h(t | x<sub>i</sub>) = h<sub>0</sub>(t) &middot; exp(&beta;<sub>1</sub> &middot; in_buffer<sub>i</sub> + &beta;<sub>2</sub> &middot; pre_mean_ntl<sub>i</sub> + &gamma; &middot; NLCD<sub>i</sub>)</div>
            <div class="formula__caption">h<sub>0</sub>(t) = nonparametric baseline hazard; HR = exp(&beta;<sub>1</sub>); HR &gt; 1 = faster recovery</div>
          </div>
          <p>
            The first three models flatten the post-disaster window into a single average. Cox
            models the <em>time to recovery</em> explicitly: buffer pixels recover
            <strong>~13% faster</strong> (HR = 1.13, p &lt; 0.001), and this holds across
            80%/90%/95% thresholds (HR = 1.12–1.13). This is the most parameter-stable result
            across all four models.
          </p>
          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.cox }">
            <div class="collapsible__content">
              <div class="data-table">
                <table>
                  <thead><tr><th>Threshold</th><th>HR</th><th>p-value</th></tr></thead>
                  <tbody>
                    <tr><td>80% of BAU</td><td class="mono" style="color:var(--green)">1.133</td><td class="mono">&lt;0.001</td></tr>
                    <tr><td>90% of BAU</td><td class="mono" style="color:var(--green)">1.126</td><td class="mono">&lt;0.001</td></tr>
                    <tr><td>95% of BAU</td><td class="mono" style="color:var(--green)">1.123</td><td class="mono">&lt;0.001</td></tr>
                  </tbody>
                </table>
              </div>
              <div class="callout callout--amber">
                <span>⚠️</span>
                <div>
                  <strong>PH assumption caveat:</strong> Schoenfeld residual tests show PH holds
                  for <code>in_buffer</code> (p = 0.82) but is severely violated for event dummies
                  (Irma p &asymp; 10<sup>-75</sup>). Miami recovers in days; Maria takes months.
                  The global HR = 1.13 is an average — not a constant across all time points.
                  Reported as a limitation.
                </div>
              </div>
              <p>
                The Kaplan-Meier curves show: early (0–10 days) both groups are similarly affected;
                mid-period (10–30 days) buffer pixels pull ahead; late (&gt;30 days) both converge.
                The generator advantage manifests as <em>recovery speed</em>, not <em>final outcome</em>.
              </p>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.cox" />
            <button class="collapsible__toggle" @click="codeExpanded.cox = !codeExpanded.cox">
              {{ codeExpanded.cox ? 'Collapse Details' : 'Show Full Cox Results' }}
            </button>
          </div>

          <!-- ═══ 5.7 Land-use confound ═══ -->
          <h2 id="sec-5-7">4.7 The Land-Use Confound</h2>
          <p>
            The most important methodological finding: when NLCD land-use controls are added,
            the buffer coefficient <strong>reverses sign or loses significance</strong> in all models.
            Critical facilities tend to be in more developed areas (NLCD 22–24), and developed
            areas recover faster regardless of generators.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Model</th><th>no_nlcd</th><th>with_nlcd</th><th>Change</th></tr></thead>
              <tbody>
                <tr><td>OLS coef</td><td class="mono" style="color:var(--green)">+0.028</td><td class="mono" style="color:#ff6b6b">-0.024</td><td>Sign reversal</td></tr>
                <tr><td>Logit OR</td><td class="mono" style="color:var(--green)">0.68</td><td class="mono" style="color:#ff6b6b">1.18</td><td>Complete reversal</td></tr>
                <tr><td>Cox HR</td><td class="mono" style="color:var(--green)">1.13</td><td class="mono">1.05</td><td>Attenuated (still &gt;1)</td></tr>
                <tr><td>Interaction (with_nlcd)</td><td colspan="2" class="mono" style="color:var(--green)">+0.010 (p=0.0002)</td><td>Generator signal &times; brightness</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            However, the <strong>interaction term</strong> remains highly significant with NLCD:
            the generator effect is real but <em>conditional on baseline brightness</em>. In bright
            urban areas, generators produce a visible NTL bump. In dim small-city areas, the
            signal is buried in the floor effect. This is not "the effect doesn't exist" — it's
            "the effect is heterogeneous."
          </p>

          <!-- ═══ Summary ═══ -->
          <div class="takeaway">
            <div class="takeaway__label">TRIANGULATION SUMMARY</div>
            <p class="takeaway__text">
              <strong>Signal is real:</strong> All four models show significant buffer effects
              in the baseline specification (direction 100% consistent).<br /><br />
              <strong>Signal is confounded:</strong> Land-use absorbs the main effect — but the
              interaction (generator &times; brightness) survives with p = 0.0002.<br /><br />
              <strong>Signal is modest:</strong> +2.8% less decline, 32% lower damage odds,
              13% faster recovery. Small relative to pixel noise, but consistent.<br /><br />
              <strong>Generalization fails:</strong> LOEO AUC = 0.455. Linear models cannot
              transport across events — this motivates Phase 3's tree-based predictive approach,
              using <code>in_buffer</code> as a weak-supervision label (validated by 100%
              LOEO sign consistency).
            </p>
          </div>
        </template>

        <!-- 06 Feature Engineering -->
        <template v-if="sectionId === 'features'">
          <h2 id="sec-6-floor">5.1 From Findings to Features</h2>
          <p>
            The interpretive modeling phase uncovered several key patterns that directly
            shape how we engineer features for prediction:
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Interpretive Finding</th><th>Predictive Feature(s)</th></tr></thead>
              <tbody>
                <tr>
                  <td><strong>Floor effect</strong> — facilities in darker areas appear less resilient because NTL can't drop much further</td>
                  <td class="mono">below_city_median, below_median_x_group</td>
                </tr>
                <tr>
                  <td><strong>City size matters</strong> — large cities (Miami) vs small cities (Lake Charles) show systematically different resilience patterns</td>
                  <td class="mono">city_size_code, log_city_pre_mean, ntl_relative</td>
                </tr>
                <tr>
                  <td><strong>Facility type variation</strong> — hospitals and airports show stronger signals than fire stations and police</td>
                  <td class="mono">fac_group, ntl_x_group</td>
                </tr>
                <tr>
                  <td><strong>Land-use confounding</strong> — NLCD categories partially explain the buffer effect (commercial land use ≈ facilities)</td>
                  <td class="mono">near_excluded (controls for non-signal facility types)</td>
                </tr>
                <tr>
                  <td><strong>Spatial proximity is informative</strong> — distance to nearest facility predicts buffer membership</td>
                  <td class="mono">log_dist</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            Without the interpretive phase, we would have built a naive feature set that ignores
            floor effects and city-size confounds — leading to a model that works in Miami but
            fails in Lake Charles. Each feature below has a direct lineage to an interpretive
            finding.
          </p>

          <h2 id="sec-6-features">5.2 Full Feature Set — 17 features</h2>
          <div class="feature-grid">
            <div v-for="f in features17" :key="f.name" class="feature-item">
              <div class="feature-item__name mono">{{ f.name }}</div>
              <div class="feature-item__desc">{{ f.desc }}</div>
            </div>
          </div>

          <h3>Feature Categories</h3>
          <div class="data-table">
            <table>
              <thead><tr><th>Category</th><th>Features</th><th>Purpose</th></tr></thead>
              <tbody>
                <tr><td>NTL Signal</td><td class="mono">drop_magnitude, delta_ntl, log_pre/post_ntl</td><td>Outage severity and baseline brightness</td></tr>
                <tr><td>Spatial</td><td class="mono">log_dist, ntl_relative, log_city_pre_mean</td><td>Distance to facility, urban context</td></tr>
                <tr><td>Facility</td><td class="mono">near_fire/police, near_excluded, fac_group</td><td>Facility type indicators</td></tr>
                <tr><td>Controls</td><td class="mono">city_size_code, is_hurricane/earthquake</td><td>Event-level confounders</td></tr>
                <tr><td>Interactions</td><td class="mono">ntl_x_group, below_median_x_group</td><td>Floor-effect correction</td></tr>
              </tbody>
            </table>
          </div>
        </template>

        <!-- 05 Models -->
        <template v-if="sectionId === 'models'">

          <h2 id="sec-5-intro">6.1 From Interpretation to Prediction</h2>
          <p>
            The interpretive phase (Stage 1) validated <code>in_buffer</code> as a meaningful
            weak-supervision label: buffer pixels consistently show higher resilience ratios across
            all events. Now we flip the direction: instead of testing whether buffer zones are
            resilient, we <strong>predict which pixels have backup power</strong> from their NTL
            behavior and spatial context.
          </p>
          <p>
            We design <strong>four model variants (A–D)</strong> as a systematic ablation study.
            Each variant removes a category of features to isolate what drives prediction accuracy.
            All four are evaluated with <strong>Leave-One-Event-Out (LOEO)</strong> cross-validation
            across 25 disaster events — ensuring that the model never trains and tests on the same
            geographic area.
          </p>

          <div class="data-table">
            <table>
              <thead><tr><th>Model</th><th>Features</th><th>Hypothesis Tested</th><th>LOEO AUC (RF)</th></tr></thead>
              <tbody>
                <tr><td style="font-weight:600">Model A</td><td>All 17 features</td><td>Full predictive power with spatial + NTL</td><td class="mono" style="color:var(--green)">0.969</td></tr>
                <tr><td style="font-weight:600">Model B</td><td>Remove pre-disaster NTL</td><td>Is post-disaster behavior alone sufficient?</td><td class="mono" style="color:var(--green)">0.970</td></tr>
                <tr><td style="font-weight:600">Model C</td><td>Model A + building footprints</td><td>Does OSM building coverage add signal?</td><td class="mono" style="color:var(--green)">0.968</td></tr>
                <tr><td style="font-weight:600">Model D</td><td>Pure NTL behavior only</td><td>Can lights alone detect generators?</td><td class="mono" style="color:var(--cyan)">0.700</td></tr>
              </tbody>
            </table>
          </div>

          <h2 id="sec-5-algo">6.2 Model A — Full Feature Set</h2>
          <p>
            Model A is the primary model, using all <strong>17 engineered features</strong> spanning
            four categories: NTL behavior (6 features), spatial proximity (4), city/disaster controls
            (3), and interaction terms (4). Three algorithms are compared:
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Algorithm</th><th>Key Hyperparameters</th><th>Role</th></tr></thead>
              <tbody>
                <tr><td>Random Forest</td><td class="mono">n=200, max_depth=8, min_samples_leaf=20, balanced</td><td>Primary classifier</td></tr>
                <tr><td>XGBoost</td><td class="mono">n=200, max_depth=5, lr=0.05, early stopping</td><td>Gradient boosting</td></tr>
                <tr><td>Logistic Regression</td><td class="mono">C=1.0, class_weight=balanced</td><td>Linear baseline</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            The final ensemble combines RF and XGBoost: <code>P = 0.7 × P_RF + 0.3 × P_XGB</code>.
            RF receives higher weight because it shows more consistent cross-event performance.
          </p>
          <div class="formula-block">
            <div class="formula">LOEO AUC: RF = 0.969 (±0.023), XGB = 0.973, Logit = 0.948</div>
            <div class="formula__caption">Mean across 25 held-out events, strict buffer label</div>
          </div>
          <p>
            Feature importance reveals <strong><code>log_dist</code></strong> (distance to nearest
            facility) as the dominant predictor (39% average importance), followed by facility type
            indicators. NTL behavior features (<code>log_pre_ntl</code>, <code>delta_ntl</code>)
            contribute modestly but consistently.
          </p>

          <h3>Why AUC, Not Precision/Recall?</h3>
          <p>
            Classification metrics like precision, recall, and F1 require choosing a probability
            threshold (e.g., P > 0.5 = "has generator"). But our goal is not binary classification
            — it's <strong>spatial ranking</strong>: which areas are more likely to have backup power?
            The threshold choice is arbitrary and application-dependent.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Metric</th><th>Model A (RF)</th><th>Why It Matters (or Doesn't)</th></tr></thead>
              <tbody>
                <tr><td style="font-weight:600">LOEO AUC</td><td class="mono" style="color:var(--green)">0.969</td>
                  <td>Threshold-free. Measures whether the model ranks generator areas above non-generator areas in <em>unseen cities</em>. This is what we care about.</td></tr>
                <tr><td style="font-weight:600">PR-AUC</td><td class="mono" style="color:var(--green)">0.949</td>
                  <td>More informative than ROC-AUC when classes are imbalanced (22% positive). High PR-AUC confirms the model isn't just predicting "no generator" everywhere.</td></tr>
                <tr><td>Precision @0.5</td><td class="mono">0.822</td>
                  <td style="color:var(--text-muted)">Depends on arbitrary threshold. Would change entirely at 0.3 or 0.7.</td></tr>
                <tr><td>Recall @0.5</td><td class="mono">0.887</td>
                  <td style="color:var(--text-muted)">Same issue. High recall at 0.5 but meaningless without context of threshold choice.</td></tr>
                <tr><td>F1 @0.5</td><td class="mono">0.853</td>
                  <td style="color:var(--text-muted)">Harmonic mean of two threshold-dependent metrics. Reported for completeness only.</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            An AUC of 0.969 means: if you randomly pick one pixel from a generator area and one
            from a non-generator area, the model assigns a higher probability to the generator
            pixel <strong>96.9% of the time</strong>. This holds across 15 completely unseen cities
            — a strong indicator of genuine cross-city generalization.
          </p>

          <h3>6.3 Model B — Post-Disaster Only</h3>
          <p>
            Model B removes all pre-disaster NTL features (<code>log_pre_ntl</code>,
            <code>ntl_relative</code>, <code>log_city_pre_mean</code>, <code>below_city_median</code>).
            This tests whether the generator detection signal comes from <strong>post-disaster
            behavior</strong> (lights staying on during outage) or <strong>pre-disaster urban
            structure</strong> (brighter areas = more infrastructure).
          </p>
          <div class="formula-block">
            <div class="formula">Model B AUC = 0.970 vs Model A AUC = 0.969 → Delta = +0.001</div>
            <div class="formula__caption">Pre-disaster brightness contributes negligibly</div>
          </div>
          <p>
            The near-identical AUC confirms that <strong>pre-disaster NTL is not necessary</strong>
            for prediction. The model relies primarily on spatial proximity features and post-disaster
            NTL changes. This is methodologically important: it means the model is not simply learning
            "bright areas have generators" but detecting genuine behavioral signals.
          </p>

          <h3>6.4 Model C — With Building Footprints</h3>
          <p>
            Model C augments Model A with four features derived from <strong>OSM building footprint
            coverage</strong> within each pixel: total coverage ratio, log-transformed coverage,
            coverage × pre-NTL interaction, and a binary indicator for meaningful building presence
            (>1% coverage).
          </p>
          <div class="formula-block">
            <div class="formula">Model C AUC = 0.968 vs Model A AUC = 0.969 → Delta = -0.001</div>
            <div class="formula__caption">Building footprints do not improve prediction</div>
          </div>
          <p>
            The null result makes sense: building footprints correlate strongly with existing features
            (<code>log_pre_ntl</code>, <code>log_dist</code>) and add no independent signal. At 500m
            resolution, individual building outlines are too fine-grained to help pixel-level prediction.
          </p>

          <h3>6.5 Model D — Pure NTL Behavior</h3>
          <p>
            Model D is the most restrictive ablation: it removes <strong>all spatial proximity
            features</strong> (<code>log_dist</code>, <code>near_fire_station</code>,
            <code>near_police</code>, <code>near_excluded</code>, <code>fac_group</code>) and
            all interaction terms. Only 10 features remain, all derived from NTL magnitude
            and change.
          </p>
          <div class="formula-block">
            <div class="formula">Model D AUC = 0.700 vs Model A AUC = 0.969 → Delta = -0.269</div>
            <div class="formula__caption">Spatial features contribute +0.269 AUC</div>
          </div>
          <p>
            This is the key finding: <strong>pure nighttime light behavior achieves AUC 0.700</strong>
            — significantly above random (0.5), confirming that NTL changes carry a genuine backup
            power detection signal. But spatial context (knowing where facilities are) adds +0.269 AUC,
            nearly doubling the model's discriminative power.
          </p>
          <div class="callout callout--cyan">
            <span>--</span>
            <div>
              <strong>Interpretation:</strong> The 0.700 AUC from Model D represents the "pure remote
              sensing" capability — detecting generators solely from satellite observations without any
              ground-truth facility locations. This is the answer to the core research question: yes,
              nighttime light changes can detect backup generators from space, but spatial context
              dramatically improves accuracy.
            </div>
          </div>

          <h2 id="sec-5-loeo">6.6 LOEO Cross-Validation Design</h2>
          <p>
            Standard k-fold cross-validation is invalid for spatially autocorrelated disaster data:
            pixels within the same event are highly correlated, and random splitting would leak
            spatial information. LOEO addresses this by holding out <strong>entire events</strong>:
          </p>
          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.loeo }">
            <div class="collapsible__content">
              <div class="code-block">
                <div class="code-block__header"><span class="mono">Python · LOEO cross-validation</span></div>
                <pre><code>for held_out in events:  # 25 events
    train = df[df.event_id != held_out]  # 14 events
    test  = df[df.event_id == held_out]  # 1 event (unseen city)

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    # Test on completely unseen geographic area
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    # → Tests cross-city generalization, not just spatial interpolation</code></pre>
              </div>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.loeo" />
            <button class="collapsible__toggle" @click="codeExpanded.loeo = !codeExpanded.loeo">
              {{ codeExpanded.loeo ? 'Collapse Code' : 'Expand Full Code' }}
            </button>
          </div>
          <p>
            With 25 events spanning 19 U.S. states, 2 disaster types (hurricane, earthquake),
            and 3 city size categories (large/medium/small), LOEO tests whether a model trained on
            Miami and New Orleans can predict Jacksonville and Atlanta — a genuinely challenging
            generalization task.
          </p>

          <h2>6.7 Ground Truth Validation — Miami-Dade Generator Permits</h2>
          <p>
            To validate the model's predictions against real-world data, we obtained
            <strong>building permit records</strong> from Miami-Dade County that identify
            properties with generator installations. Of 592 permits county-wide, 148 fall
            within our Irma study area (30 commercial, 118 residential).
          </p>

          <div class="eda-chart-card" style="text-align:center">
            <img :src="`${base}data/miami_generator_validation.png`"
                 alt="Miami probability map with generator permit locations"
                 style="width:100%; max-width:700px; border-radius:var(--radius)" />
            <p style="font-size:11px; color:var(--text-dim); margin-top:8px">
              Yellow diamonds = commercial generator permits, orange dots = residential.
              Background: Model A predicted backup power probability.
            </p>
          </div>

          <p>
            <strong>Commercial generators</strong> (yellow diamonds) show a statistically
            significant correlation with predicted probability: Mann-Whitney rank correlation
            = 0.684 (p = 0.0005). These are large institutional units at hospitals, hotels,
            and office buildings — exactly the type of generator that would change a 500m pixel's
            brightness.
          </p>
          <p>
            <strong>Residential generators</strong> (orange dots) show no correlation (rank = 0.340,
            p = 0.41). This is expected: a single home generator produces too little light to
            detectably change a 25-hectare pixel's total radiance. This confirms the
            <strong>500m resolution limitation</strong> — the method detects aggregate
            infrastructure-scale backup power, not individual household units.
          </p>

          <div class="callout callout--amber">
            <span>--</span>
            <div>
              <strong>Data caveat:</strong> Miami-Dade permits capture only post-construction
              generator installations, not generators built as part of original construction.
              Major facilities (airport, hospital, port) likely had generators from the start
              and are absent from permit records. The true commercial generator count is
              therefore higher than 30, making the validation conservative.
            </div>
          </div>

          <div class="takeaway">
            <div class="takeaway__label">KEY FINDINGS</div>
            <p class="takeaway__text">
              <strong>Model A achieves 0.969 mean LOEO AUC</strong> across 25 held-out events —
              strong cross-city generalization. <strong>Pre-NTL features are unnecessary</strong>
              (Model B matches Model A). <strong>Building footprints add nothing</strong> (Model C).
              <strong>Pure NTL behavior gives 0.700 AUC</strong> (Model D) — a genuine but modest
              remote sensing signal that spatial context amplifies by +0.269. Ground truth validation
              with Miami-Dade generator permits confirms that <strong>commercial generators are
              detectable</strong> (rank = 0.684) while <strong>residential generators are not</strong>
              — a physical limitation of 500m resolution.
            </p>
          </div>

          <h2>6.8 Probability Maps</h2>
          <p>
            The final ensemble model generates <code>P(backup_power_present)</code> for every
            urban pixel in each study area:
          </p>
          <div class="formula-block">
            <div class="formula">P_ensemble = 0.7 × P_RF + 0.3 × P_XGB</div>
            <div class="formula__caption">RF receives higher weight due to more consistent cross-event performance</div>
          </div>
          <p>
            Maps are exported as GeoTIFF (for analysis) and GeoJSON (for the interactive dashboard).
            The heatmap uses per-event quantile normalization (P10/P50/P90) to ensure the full color
            range is utilized regardless of the event's absolute probability distribution.
          </p>

          <RouterLink to="/map" class="feature-link reveal">
            <span class="feature-link__text">Explore the probability maps for all 25 events on the interactive map</span>
            <span class="feature-link__cta">Open Map →</span>
          </RouterLink>
        </template>

        <!-- 07 Zip-Code Analysis -->
        <template v-if="sectionId === 'stage3'">
          <h2 id="sec-7-1">7.1 Research Question</h2>
          <p>
            Stage 2 answers "can we detect backup generators from space?" at the <strong>pixel level</strong>.
            Stage 3 asks a policy-relevant follow-up: <strong>do areas with more critical facilities
            experience less severe power outages historically?</strong> We shift from 500m pixels to
            zip-code-level analysis, connecting facility density with EAGLE-I outage records.
          </p>

          <h2 id="sec-7-2">7.2 Data Sources</h2>
          <div class="data-table">
            <table>
              <thead><tr><th>Dataset</th><th>Source</th><th>Role</th></tr></thead>
              <tbody>
                <tr><td>Power outages</td><td class="mono">EAGLE-I (2014–2023)</td><td>Dependent variable: outage severity</td></tr>
                <tr><td>Facility density</td><td class="mono">OSM Overpass API</td><td>Key independent variable</td></tr>
                <tr><td>Backup power prob.</td><td class="mono">Stage 2 Model C output</td><td>Aggregated to zip code level</td></tr>
                <tr><td>Hurricane tracks</td><td class="mono">IBTrACS v4</td><td>Wind exposure + track-weighted outage allocation</td></tr>
                <tr><td>Demographics</td><td class="mono">Census ACS</td><td>Population density, income controls</td></tr>
                <tr><td>Land use</td><td class="mono">NLCD</td><td>Urban fraction control</td></tr>
                <tr><td>Zip boundaries</td><td class="mono">ZCTA shapefiles</td><td>Spatial unit</td></tr>
                <tr><td>ZIP-County crosswalk</td><td class="mono">HUD USPS</td><td>County outage → zip allocation</td></tr>
              </tbody>
            </table>
          </div>

          <h2 id="sec-7-3">7.3 Sample Construction</h2>
          <p>
            The analysis covers <strong>1,002 zip codes</strong> across <strong>22 U.S. disaster
            events</strong> (Puerto Rico and Turkey excluded due to lack of U.S. ZCTA boundaries).
            Events selected from EAGLE-I based on severity (duration > 72h, peak > 100K affected,
            >= 5 counties). Geographic coverage spans 19 U.S. states from Florida to Washington.
          </p>
          <p>
            County-level outage data is allocated to zip codes using a <strong>multi-factor
            weighting scheme</strong>: population density (Census), NTL drop magnitude
            (Stage 2 pixel panel), and wind field exposure (IBTrACS Holland model for
            hurricane events). This is more accurate than simple area-weighted HUD crosswalk
            allocation.
          </p>

          <h2 id="sec-7-4">7.4 Model Design</h2>
          <p>
            We test whether facility density predicts backup power probability at the zip code
            level, using a four-step modeling approach that mirrors the interpretive modeling
            philosophy from Stage 1: start simple, then progressively address statistical concerns.
          </p>

          <h3>Dependent Variable</h3>
          <p>
            For each zip code within a study area, we compute <code>mean_backup_prob</code> — the
            average predicted backup power probability from Stage 2's ensemble model, aggregated
            from all pixels within the zip code boundary via spatial join with ZCTA shapefiles.
          </p>

          <h3>Independent Variables</h3>
          <div class="data-table">
            <table>
              <thead><tr><th>Variable</th><th>Source</th><th>Role</th></tr></thead>
              <tbody>
                <tr><td><strong><code>facility_density</code></strong></td><td>OSM / ZCTA area</td><td>Core predictor: critical facilities per km²</td></tr>
                <tr><td><code>mean_backup_prob</code></td><td>Stage 2 Model C</td><td>Pixel-level probability aggregated to zip code</td></tr>
                <tr><td><code>ntl_drop_pct</code></td><td>pixel_panel</td><td>Mean NTL decline during disaster</td></tr>
                <tr><td><code>pop_density</code></td><td>Census ACS</td><td>Control: urbanization proxy</td></tr>
                <tr><td><code>median_income</code></td><td>Census ACS</td><td>Control: socioeconomic status</td></tr>
                <tr><td><code>urban_pct</code></td><td>NLCD</td><td>Control: land cover classification</td></tr>
                <tr><td><code>impervious_pct</code></td><td>NLCD</td><td>Control: building/road density proxy</td></tr>
                <tr><td><code>disaster_exposure</code></td><td>IBTrACS</td><td>Control: historical hurricane track density</td></tr>
              </tbody>
            </table>
          </div>

          <h3>Model 1: OLS with Event Fixed Effects</h3>
          <div class="formula-block">
            <div class="formula">mean_prob<sub>iz</sub> = &beta;<sub>0</sub> + &beta;<sub>1</sub> &middot; fac_density<sub>z</sub> + &gamma;<sub>i</sub> + &epsilon;<sub>iz</sub></div>
            <div class="formula__caption">i = event, z = zip code, &gamma;<sub>i</sub> = event fixed effect</div>
          </div>
          <p>
            The baseline model tests the core relationship while absorbing event-level variation
            (different cities, disaster types, severity) through fixed effects. This ensures we are
            comparing zip codes <em>within</em> the same disaster, not across events with different
            baseline brightness levels.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody>
                <tr><td>N</td><td class="mono">1,002 zip-event observations</td></tr>
                <tr><td>R²</td><td class="mono" style="color:var(--green)">0.475</td></tr>
                <tr><td>fac_density coefficient</td><td class="mono">+0.096 (se = 0.006)</td></tr>
                <tr><td>fac_density p-value</td><td class="mono" style="color:var(--green)">9.56 × 10⁻⁵⁴</td></tr>
                <tr><td>Interpretation</td><td>+1 facility/km² → +9.6% backup power probability</td></tr>
              </tbody>
            </table>
          </div>

          <h3>Model 2: Moran's I Spatial Autocorrelation Test</h3>
          <p>
            OLS assumes independent observations, but neighboring zip codes are spatially
            correlated — a hospital's backup power benefits adjacent zip codes too. Moran's I
            tests whether OLS residuals cluster spatially:
          </p>
          <div class="formula-block">
            <div class="formula">Moran's I = 0.329, E[I] = -0.007, p = 0.001</div>
            <div class="formula__caption">Tested on Newark NJ (n=145 zips), KNN weights (k=5)</div>
          </div>
          <p>
            <strong>Significant spatial autocorrelation</strong> — OLS standard errors are
            underestimated. We need a spatial regression model to correct for this.
          </p>

          <h3>Model 3: Spatial Error Model (SEM)</h3>
          <p>
            The Spatial Error Model accounts for spatially correlated error terms. If a zip code's
            residual is positive (model under-predicts), its neighbors' residuals tend to be positive
            too — captured by the spatial autoregressive parameter &lambda;:
          </p>
          <div class="formula-block">
            <div class="formula">y = X&beta; + u, &nbsp; u = &lambda;Wu + &epsilon;</div>
            <div class="formula__caption">W = spatial weights matrix (KNN, k=5), &lambda; = spatial error parameter</div>
          </div>
          <div class="data-table">
            <table>
              <thead><tr><th>Parameter</th><th>Value</th><th>Interpretation</th></tr></thead>
              <tbody>
                <tr><td>&lambda;</td><td class="mono" style="color:var(--cyan)">0.648</td><td>Strong spatial error dependence — neighbors' errors are 65% correlated</td></tr>
                <tr><td>fac_density</td><td class="mono" style="color:var(--green)">0.097</td><td>Effect <strong>survives</strong> spatial correction (virtually unchanged from OLS)</td></tr>
                <tr><td>Pseudo R²</td><td class="mono">0.236</td><td>Lower than OLS R² because spatial effects absorb variance</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            The key finding: <strong>facility density's effect is real, not a spatial artifact</strong>.
            The coefficient barely changes from OLS (0.096 → 0.097) after controlling for spatial
            autocorrelation. The high &lambda; (0.648) means neighboring zip codes share unobserved
            factors (urban form, utility infrastructure), but the facility density signal is
            independent of this spatial structure.
          </p>

          <h3>Model 4: Subgroup Analysis</h3>
          <p>
            Does the facility density effect vary by city size or disaster type? This connects
            to the floor effect discovered in Stage 1 — smaller cities might show different patterns.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>City Size</th><th>N</th><th>R²</th><th>With Facilities</th><th>Without</th><th>Delta</th></tr></thead>
              <tbody>
                <tr><td>Large</td><td class="mono">687</td><td class="mono">0.358</td><td class="mono">0.480</td><td class="mono">0.310</td><td class="mono" style="color:var(--green)">+0.170</td></tr>
                <tr><td>Medium</td><td class="mono">241</td><td class="mono">0.284</td><td class="mono">0.271</td><td class="mono">0.129</td><td class="mono" style="color:var(--green)">+0.142</td></tr>
                <tr><td>Small</td><td class="mono">74</td><td class="mono">0.326</td><td class="mono">0.229</td><td class="mono">0.064</td><td class="mono" style="color:var(--green)">+0.165</td></tr>
              </tbody>
            </table>
          </div>
          <div class="data-table" style="margin-top:12px">
            <table>
              <thead><tr><th>Disaster Type</th><th>N</th><th>With Facilities</th><th>Without</th><th>Delta</th></tr></thead>
              <tbody>
                <tr><td>Hurricane</td><td class="mono">554</td><td class="mono">0.403</td><td class="mono">0.233</td><td class="mono" style="color:var(--green)">+0.171</td></tr>
                <tr><td>Winter Storm</td><td class="mono">234</td><td class="mono">0.435</td><td class="mono">0.274</td><td class="mono" style="color:var(--green)">+0.161</td></tr>
                <tr><td>Derecho</td><td class="mono">74</td><td class="mono">0.534</td><td class="mono">0.263</td><td class="mono" style="color:var(--green)">+0.271</td></tr>
                <tr><td>Severe Storm</td><td class="mono">90</td><td class="mono">0.397</td><td class="mono">0.127</td><td class="mono" style="color:var(--green)">+0.270</td></tr>
                <tr><td>Ice Storm</td><td class="mono">50</td><td class="mono">0.296</td><td class="mono">0.077</td><td class="mono" style="color:var(--green)">+0.219</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            The effect is <strong>universally positive</strong> across all city sizes and all
            disaster types — no exceptions. Derechos and severe storms show the largest deltas
            (+0.27), possibly because these events cause more localized, patchy outages where
            generator-powered facilities stand out more clearly against the surrounding blackout.
          </p>

          <div class="takeaway">
            <div class="takeaway__label">KEY FINDINGS</div>
            <p class="takeaway__text">
              Facility density is a <strong>robust predictor of backup power probability</strong>
              at the zip code level: OLS R² = 0.475 (p = 9.6e-54), effect survives spatial
              error correction (&lambda; = 0.648), and is consistent across all city sizes
              and all 5 disaster types. The +9.6% per facility/km² effect translates to
              meaningful policy insight: zip codes with denser critical infrastructure have
              measurably higher predicted resilience during power outages.
            </p>
          </div>

        </template>

        <!-- Conclusions & Future Work -->
        <template v-if="sectionId === 'conclusions'">
          <h2 id="sec-c-1">Conclusions</h2>
          <p>
            Across three stages of analysis — interpretive modeling, predictive modeling, and
            zip-code spatial regression — several conclusions emerge:
          </p>
          <ul class="detail-list">
            <li><strong>The satellite signal is real but modest.</strong> Pure nighttime light
              behavior (Model D) achieves AUC 0.704 — above random, confirming that brightness
              anomalies during outages carry genuine information about backup power. But this
              signal alone is insufficient for reliable detection at 500m resolution.</li>
            <li><strong>Spatial context is the dominant predictor.</strong> Knowing where critical
              facilities are located adds +0.263 AUC, raising performance to 0.967. The model
              primarily learns that areas near hospitals and airports tend to stay brighter — a
              useful but less novel finding.</li>
            <li><strong>The effect generalizes across geography and disaster type.</strong> 25/25
              events show consistent positive resilience advantage for facility buffer zones.
              The zip-code analysis confirms this at a coarser scale: 22/22 U.S. events, all
              5 disaster types, all 3 city size categories.</li>
            <li><strong>Commercial generators are detectable; residential are not.</strong>
              Miami-Dade ground truth validation shows the method works for large institutional
              generators (rank = 0.684) but cannot resolve individual household units at 500m.</li>
            <li><strong>No unified generator database exists, and building one from permits is
              impractical.</strong> This data gap — confirmed through our permit collection effort
              — is itself the strongest motivation for satellite-based approaches.</li>
          </ul>

          <h2 id="sec-c-2">Limitations</h2>
          <ul class="detail-list">
            <li><strong>Resolution.</strong> 500m pixels mix multiple land uses; the method detects
              neighborhood-level patterns, not individual buildings. Higher-resolution sensors
              (e.g., Jilin-1, EROS) could improve building-level detection but lack daily global
              coverage.</li>
            <li><strong>Weak supervision label.</strong> We assume facilities near OSM-listed
              infrastructure have generators — a reasonable but imperfect proxy. Some facilities
              may lack generators; some generators exist at unlisted locations.</li>
            <li><strong>Gap-filled imagery.</strong> The expanded 25-event dataset uses NASA's
              gap-filled VNP46A2, where cloudy days are interpolated from neighbors. This may
              slightly attenuate the outage signal during storm periods.</li>
            <li><strong>Temporal resolution.</strong> Daily imagery captures overnight brightness
              but misses daytime generator operation, load fluctuations, and generators that only
              run during peak demand hours.</li>
          </ul>

          <h2 id="sec-c-3">Future Directions</h2>
          <ul class="detail-list">
            <li><strong>Higher-resolution imagery.</strong> Commercial satellite constellations
              now offer sub-meter nighttime imagery. Combining VIIRS's daily temporal coverage
              with occasional high-resolution snapshots could bridge the resolution gap.</li>
            <li><strong>Integration with utility data.</strong> Where smart meter or SCADA data
              is available, supervised learning with actual outage/restoration timestamps could
              replace the weak-supervision buffer approach.</li>
            <li><strong>Emission estimation.</strong> Detected generator locations could be combined
              with fuel type databases and runtime estimates to model backup generator emissions
              — relevant for environmental justice and air quality assessment.</li>
            <li><strong>Real-time monitoring.</strong> As NTL products approach near-real-time
              delivery, the method could support emergency response: identifying which areas
              have activated backup power within hours of a blackout.</li>
            <li><strong>Cross-country extension.</strong> The method is sensor-agnostic and
              transferable. Applying it to non-U.S. disasters (where outage records are even
              scarcer) could provide first-of-its-kind resilience mapping in data-poor regions.</li>
          </ul>
        </template>

        <!-- 09 Dashboard Development -->
        <template v-if="sectionId === 'web'">
          <h2 id="sec-8-1">9.1 Architecture</h2>
          <p>
            This dashboard is a single-page application built with <strong>Vue 3</strong> (Composition API)
            and <strong>Vite</strong> as the build tool. The project uses Vue Router (hash history mode
            for GitHub Pages compatibility) with lazy-loaded route components to minimize initial bundle size.
          </p>
          <div class="data-table">
            <table>
              <thead><tr><th>Technology</th><th>Role</th></tr></thead>
              <tbody>
                <tr><td class="mono">Vue 3 + Vite</td><td>SPA framework + dev/build toolchain</td></tr>
                <tr><td class="mono">Vue Router</td><td>Client-side routing (hash mode)</td></tr>
                <tr><td class="mono">MapLibre GL JS</td><td>WebGL map rendering engine</td></tr>
                <tr><td class="mono">GitHub Actions</td><td>CI/CD pipeline for deployment</td></tr>
                <tr><td class="mono">GitHub Pages</td><td>Static hosting</td></tr>
              </tbody>
            </table>
          </div>

          <h2 id="sec-8-2">9.2 Map Engine</h2>
          <p>
            The interactive map uses <strong>MapLibre GL JS</strong> with multiple layer types per event:
          </p>
          <ul class="detail-list">
            <li><strong>Heatmap layer</strong> — per-event quantile-normalized probability overlay with adaptive color ramp</li>
            <li><strong>Symbol layer</strong> — canvas-rendered facility icons (hospital, airport, fire station, etc.)</li>
            <li><strong>Circle layer</strong> — invisible hit targets for pixel-level probability tooltips</li>
            <li><strong>Fill + line layers</strong> — optional buffer zone visualization</li>
            <li><strong>Overview markers</strong> — colored dots with labels at low zoom, transitioning to detail layers at zoom >= 8</li>
          </ul>
          <p>
            Basemap options include CARTO Dark Matter, Positron, Voyager, and ESRI World Imagery satellite.
            Event data is <strong>lazy-loaded</strong> on demand — only the selected event's GeoJSON is fetched,
            keeping initial page load fast even with 25 events.
          </p>

          <h2 id="sec-8-3">9.3 Responsive Design</h2>
          <p>
            The dashboard adapts to mobile devices with:
          </p>
          <ul class="detail-list">
            <li><strong>NavBar</strong> — hamburger menu below 768px with slide-down navigation</li>
            <li><strong>Map page</strong> — sidebars auto-collapse on mobile, scroll-limited event panel</li>
            <li><strong>Home page</strong> — compact event chips grid, stacked CTAs, single-column pipeline</li>
            <li><strong>Docs/Charts</strong> — horizontal-scrolling tables, reduced padding, stacked layouts</li>
          </ul>

          <h2 id="sec-8-4">9.4 Deployment</h2>
          <p>
            Automated via <strong>GitHub Actions</strong>: every push to <code>main</code> that modifies
            <code>project/nightlight-dashboard/**</code> triggers a build + deploy workflow. The Vite
            build outputs to <code>dist/</code> with <code>base: '/Practicum/'</code> for correct
            GitHub Pages path resolution. Data files (GeoJSON, facility JSON) are bundled in
            <code>public/data/</code> and served statically.
          </p>
        </template>

        <!-- 10 Reproducibility -->
        <template v-if="sectionId === 'repro'">
          <div class="data-table">
            <table>
              <thead><tr><th>Resource</th><th>Details</th></tr></thead>
              <tbody>
                <tr><td>VNP46A2 daily NTL</td><td class="mono">GEE: NASA/VIIRS/002/VNP46A2</td></tr>
                <tr><td>VNP46A3 monthly NTL</td><td class="mono">GEE: NASA/VIIRS/002/VNP46A3</td></tr>
                <tr><td>EAGLE-I outage data</td><td>U.S. DOE EAGLE-I portal (public)</td></tr>
                <tr><td>Facility POI</td><td>OpenStreetMap Overpass API via <code>overpy</code></td></tr>
                <tr><td>Pipeline</td><td>stage2_prediction_model.ipynb (Python / Jupyter)</td></tr>
                <tr><td>Dashboard export</td><td><code>export_to_dashboard.py</code></td></tr>
              </tbody>
            </table>
          </div>
          <div class="callout callout--green">
            <span>📖</span>
            <div>
              <strong>Key references:</strong>
              Wang et al. (2018, NASA Black Marble team) — disaster power outage monitoring;
              Zhang et al. (2023) — damage assessment with Black Marble NTL.
            </div>
          </div>

          <h3>Environment</h3>
          <div class="collapsible collapsible--code" :class="{ expanded: codeExpanded.deps }">
            <div class="collapsible__content">
              <div class="code-block">
                <div class="code-block__header"><span class="mono">Key Dependencies</span></div>
                <pre><code>python >= 3.10
pandas, geopandas, rasterio, pyproj
scikit-learn, xgboost, joblib
overpy (OSM Overpass)
google-earth-engine (GEE API)</code></pre>
              </div>
            </div>
            <div class="collapsible__fade" v-if="!codeExpanded.deps" />
            <button class="collapsible__toggle" @click="codeExpanded.deps = !codeExpanded.deps">
              {{ codeExpanded.deps ? 'Collapse Code' : 'Expand Full Code' }}
            </button>
          </div>
        </template>

        <!-- Bottom navigation -->
        <div class="detail-nav">
          <RouterLink v-if="prevSection" :to="`/docs/${prevSection.id}`" class="detail-nav__link detail-nav__link--prev">
            <span class="detail-nav__dir">← Previous</span>
            <span class="detail-nav__name">{{ prevSection.title }}</span>
          </RouterLink>
          <div v-else />
          <RouterLink v-if="nextSection" :to="`/docs/${nextSection.id}`" class="detail-nav__link detail-nav__link--next">
            <span class="detail-nav__dir">Next →</span>
            <span class="detail-nav__name">{{ nextSection.title }}</span>
          </RouterLink>
        </div>
      </article>

      <div v-else class="not-found">
        <p>Section not found.</p>
        <RouterLink to="/docs" class="back-link">← Back to Documentation</RouterLink>
      </div>
    </div>

    <!-- ── Chart modal (click to enlarge) ── -->
    <Teleport to="body">
      <div v-if="chartModal" class="chart-modal" @click.self="chartModal = null">
        <div class="chart-modal__card">
          <button class="chart-modal__close" @click="chartModal = null">&times;</button>
          <div class="chart-modal__header">
            <span class="dot" :style="{ background: chartModal.ev.color }" />
            <strong>{{ chartModal.ev.name }}</strong>
            <span style="color:var(--text-muted)">{{ chartModal.ev.subtitle }}</span>
            <span class="mono" style="color:var(--text-dim); font-size:11px">{{ chartModal.ev.year }}</span>
          </div>

          <!-- NTL line chart (large) -->
          <template v-if="chartModal.type === 'ntl'">
            <div style="font-size:12px; color:var(--text-muted); margin-bottom:8px">
              Pre avg: {{ cloudStats[chartModal.ev.dashId]?.summary.pre_mean_ntl }} ·
              Post avg: {{ cloudStats[chartModal.ev.dashId]?.summary.post_mean_ntl }} nW/cm²/sr
            </div>
            <svg :viewBox="`0 0 ${modalW} ${modalH}`" class="ntl-svg" preserveAspectRatio="xMidYMid meet">
              <line v-for="y in [0.25, 0.5, 0.75, 1.0]" :key="y"
                :x1="modalPad.l" :x2="modalW - modalPad.r"
                :y1="modalNtlY(y, chartModal.ev.id)" :y2="modalNtlY(y, chartModal.ev.id)"
                stroke="rgba(255,255,255,0.08)" stroke-width="0.5" />
              <line :x1="modalSplitX(chartModal.ev.id)" :x2="modalSplitX(chartModal.ev.id)"
                :y1="modalPad.t" :y2="modalH - modalPad.b"
                stroke="rgba(255,100,100,0.6)" stroke-width="1.5" stroke-dasharray="4 3" />
              <text :x="modalSplitX(chartModal.ev.id) + 5" :y="modalPad.t + 14" fill="rgba(255,120,120,0.8)" font-size="11" font-family="monospace">DISASTER</text>
              <polyline :points="modalLinePath(chartModal.ev.id, 'pre')" fill="none" stroke="rgba(0,229,160,0.9)" stroke-width="2" />
              <polyline :points="modalLinePath(chartModal.ev.id, 'post')" fill="none" stroke="rgba(255,107,107,0.9)" stroke-width="2" />
              <polygon :points="modalAreaPath(chartModal.ev.id, 'pre')" fill="rgba(0,229,160,0.15)" />
              <polygon :points="modalAreaPath(chartModal.ev.id, 'post')" fill="rgba(255,107,107,0.15)" />
              <text :x="modalPad.l + 4" :y="modalH - 4" fill="rgba(0,229,160,0.8)" font-size="10" font-family="monospace">Pre-disaster</text>
              <text :x="modalSplitX(chartModal.ev.id) + 5" :y="modalH - 4" fill="rgba(255,107,107,0.8)" font-size="10" font-family="monospace">Post-disaster</text>
            </svg>
          </template>

          <!-- Cloud fraction chart (large) -->
          <template v-if="chartModal.type === 'cloud'">
            <div style="font-size:12px; color:var(--text-muted); margin-bottom:8px">
              Avg cloud: {{ cloudFrac[chartModal.ev.dashId]?.summary.avg_cloud_pct }}% ·
              {{ cloudFrac[chartModal.ev.dashId]?.summary.excluded_days }} days excluded
            </div>
            <svg :viewBox="`0 0 ${modalW} ${modalH}`" class="ntl-svg" preserveAspectRatio="xMidYMid meet">
              <line :x1="modalPad.l" :x2="modalW - modalPad.r"
                :y1="modalCloudY(30)" :y2="modalCloudY(30)"
                stroke="rgba(255,170,0,0.6)" stroke-width="1" stroke-dasharray="4 2" />
              <text :x="modalW - modalPad.r - 4" :y="modalCloudY(30) - 4" fill="rgba(255,170,0,0.8)" font-size="10" font-family="monospace" text-anchor="end">30% threshold</text>
              <line :x1="modalCfSplitX(chartModal.ev.dashId)" :x2="modalCfSplitX(chartModal.ev.dashId)"
                :y1="modalPad.t" :y2="modalH - modalPad.b"
                stroke="rgba(255,255,255,0.2)" stroke-width="1" stroke-dasharray="3 2" />
              <rect v-for="(d, i) in getCfDays(chartModal.ev.dashId)" :key="'mcf'+i"
                :x="modalCfBarX(chartModal.ev.dashId, i)" :y="modalCloudY(d.cloud_pct)"
                :width="modalCfBarW(chartModal.ev.dashId)" :height="modalH - modalPad.b - modalCloudY(d.cloud_pct)"
                :fill="d.usable ? 'rgba(0,180,255,0.55)' : 'rgba(255,80,80,0.6)'"
                rx="1" />
            </svg>
          </template>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { EVENTS } from '@/data/events.js'

const route = useRoute()
const base = import.meta.env.BASE_URL
const sectionId = computed(() => route.params.section)

// Scroll to top when navigating between doc sections
watch(sectionId, () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
  // Re-setup reveal observers for new content
  setTimeout(() => setupReveal(), 300)
})

// ── NTL Animation Player ──
const ntlFrames = ref(null)
const frameIdx = ref(0)
const playing = ref(false)
let playInterval = null

onMounted(async () => {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}data/frames/maria_frames.json`)
    if (res.ok) ntlFrames.value = await res.json()
  } catch { /* ignore */ }
})

function togglePlay() {
  if (playing.value) {
    clearInterval(playInterval)
    playing.value = false
  } else {
    playing.value = true
    playInterval = setInterval(() => {
      if (frameIdx.value < (ntlFrames.value?.frames.length ?? 1) - 1) {
        frameIdx.value++
      } else {
        frameIdx.value = 0
      }
    }, 400)
  }
}

function nextFrame() {
  if (ntlFrames.value && frameIdx.value < ntlFrames.value.frames.length - 1) frameIdx.value++
}
function prevFrame() {
  if (frameIdx.value > 0) frameIdx.value--
}

onUnmounted(() => { clearInterval(playInterval); tocObserver?.disconnect(); revealObs?.disconnect() })

// ── Collapsible state ──
const ntlExpanded = ref(false)
const chartModal = ref(null)

// Modal chart dimensions (larger)
const modalW = 720
const modalH = 240
const modalPad = { t: 20, b: 18, l: 12, r: 12 }

function modalNtlY(frac, dashId) {
  return modalPad.t + (modalH - modalPad.t - modalPad.b) * (1 - Math.min(frac, 1))
}
function modalSplitX(dashId) {
  const all = getAllDays(dashId)
  const pre = getPreDays(dashId).length
  if (!all.length) return modalPad.l
  const w = (modalW - modalPad.l - modalPad.r) / all.length
  return modalPad.l + pre * w
}
function modalLinePath(dashId, phase) {
  const days = phase === 'pre' ? getPreDays(dashId) : getPostDays(dashId)
  const total = getAllDays(dashId).length
  if (!days.length || !total) return ''
  const w = (modalW - modalPad.l - modalPad.r) / total
  const offset = phase === 'post' ? getPreDays(dashId).length : 0
  const mx = ntlMax(dashId)
  return days.map((d, i) => {
    const x = modalPad.l + (offset + i) * w + w / 2
    const y = modalNtlY(d.mean_ntl / mx, dashId)
    return `${x},${y}`
  }).join(' ')
}
function modalAreaPath(dashId, phase) {
  const days = phase === 'pre' ? getPreDays(dashId) : getPostDays(dashId)
  const total = getAllDays(dashId).length
  if (!days.length || !total) return ''
  const w = (modalW - modalPad.l - modalPad.r) / total
  const offset = phase === 'post' ? getPreDays(dashId).length : 0
  const mx = ntlMax(dashId)
  const baseline = modalH - modalPad.b
  const pts = days.map((d, i) => {
    const x = modalPad.l + (offset + i) * w + w / 2
    return `${x},${modalNtlY(d.mean_ntl / mx, dashId)}`
  })
  const firstX = modalPad.l + offset * w + w / 2
  const lastX = modalPad.l + (offset + days.length - 1) * w + w / 2
  return `${firstX},${baseline} ${pts.join(' ')} ${lastX},${baseline}`
}
function modalCloudY(pct) {
  return modalPad.t + (modalH - modalPad.t - modalPad.b) * (1 - pct / 100)
}
function modalCfSplitX(dashId) {
  const days = getCfDays(dashId)
  const preCount = days.filter(d => d.period === 'pre').length
  if (!days.length) return modalPad.l
  const w = (modalW - modalPad.l - modalPad.r) / days.length
  return modalPad.l + preCount * w
}
function modalCfBarW(dashId) {
  const total = getCfDays(dashId).length
  return total > 0 ? Math.max((modalW - modalPad.l - modalPad.r) / total - 1, 1) : 2
}
function modalCfBarX(dashId, i) {
  const total = getCfDays(dashId).length
  const w = (modalW - modalPad.l - modalPad.r) / total
  return modalPad.l + i * w + 0.5
}
const cloudExpanded = ref(false)
const codeExpanded = reactive({
  gee: false,
  overpass: false,
  loeo: false,
  deps: false,
  ols: false,
  mixed: false,
  logit: false,
  cloudTable: false,
  cox: false,
})

// ── Cloud / NTL stats ──
const cloudStats = ref(null)
const edaStats = ref(null)
const cloudFrac = ref(null)

onMounted(async () => {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}data/cloud_stats.json`)
    if (res.ok) cloudStats.value = await res.json()
  } catch { /* ignore */ }
  try {
    const res2 = await fetch(`${import.meta.env.BASE_URL}data/eda_stats.json`)
    if (res2.ok) edaStats.value = await res2.json()
  } catch { /* ignore */ }
  try {
    const res3 = await fetch(`${import.meta.env.BASE_URL}data/cloud_fraction.json`)
    if (res3.ok) cloudFrac.value = await res3.json()
  } catch { /* ignore */ }
})

const DASH_MAP = {
  maria: 'maria', irma: 'irma', ida: 'ida', laura: 'laura',
  michael: 'michael', 'eq-pr': 'eq-pr', 'ian-charlotte': 'ian-charlotte',
  'ian-fortmyers': 'ian-fortmyers', 'eq-hatay': 'eq-hatay',
}

const cloudEvents = computed(() =>
  EVENTS.map(ev => ({ ...ev, dashId: ev.id }))
)

// ── NTL bar chart helpers ──
const ntlChartW = 480
const ntlChartH = 120
const ntlPad = { t: 16, b: 14, l: 8, r: 8 }

function getPreDays(dashId) { return cloudStats.value?.[dashId]?.pre ?? [] }
function getPostDays(dashId) { return cloudStats.value?.[dashId]?.post ?? [] }
function getAllDays(dashId) { return [...getPreDays(dashId), ...getPostDays(dashId)] }

function ntlMax(dashId) {
  const all = getAllDays(dashId)
  return all.length ? Math.max(...all.map(d => d.mean_ntl), 1) : 1
}

function ntlY(frac, dashId) {
  return ntlPad.t + (ntlChartH - ntlPad.t - ntlPad.b) * (1 - Math.min(frac, 1))
}

function ntlBarW(dashId) {
  const total = getAllDays(dashId).length
  return total > 0 ? Math.max((ntlChartW - ntlPad.l - ntlPad.r) / total - 1, 1) : 2
}

function ntlBarX(dashId, i, phase) {
  const total = getAllDays(dashId).length
  const w = (ntlChartW - ntlPad.l - ntlPad.r) / total
  const offset = phase === 'post' ? getPreDays(dashId).length : 0
  return ntlPad.l + (offset + i) * w + 0.5
}

function ntlLinePath(dashId, phase) {
  const days = phase === 'pre' ? getPreDays(dashId) : getPostDays(dashId)
  const total = getAllDays(dashId).length
  if (!days.length || !total) return ''
  const w = (ntlChartW - ntlPad.l - ntlPad.r) / total
  const offset = phase === 'post' ? getPreDays(dashId).length : 0
  const mx = ntlMax(dashId)
  return days.map((d, i) => {
    const x = ntlPad.l + (offset + i) * w + w / 2
    const y = ntlY(d.mean_ntl / mx, dashId)
    return `${x},${y}`
  }).join(' ')
}

function ntlAreaPath(dashId, phase) {
  const days = phase === 'pre' ? getPreDays(dashId) : getPostDays(dashId)
  const total = getAllDays(dashId).length
  if (!days.length || !total) return ''
  const w = (ntlChartW - ntlPad.l - ntlPad.r) / total
  const offset = phase === 'post' ? getPreDays(dashId).length : 0
  const mx = ntlMax(dashId)
  const baseline = ntlChartH - ntlPad.b
  const pts = days.map((d, i) => {
    const x = ntlPad.l + (offset + i) * w + w / 2
    const y = ntlY(d.mean_ntl / mx, dashId)
    return `${x},${y}`
  })
  const firstX = ntlPad.l + offset * w + w / 2
  const lastX = ntlPad.l + (offset + days.length - 1) * w + w / 2
  return `${firstX},${baseline} ${pts.join(' ')} ${lastX},${baseline}`
}

function ntlX(dashId, type) {
  if (type === 'split') {
    const pre = getPreDays(dashId).length
    const total = getAllDays(dashId).length
    return total > 0 ? ntlPad.l + (pre / total) * (ntlChartW - ntlPad.l - ntlPad.r) : ntlPad.l
  }
  return ntlPad.l
}

// ── Cloud chart helpers ──
function cloudY(pct) {
  return ntlPad.t + (ntlChartH - ntlPad.t - ntlPad.b) * (1 - pct / 100)
}

function cloudBarW(dashId) {
  const total = getAllDays(dashId).length
  return total > 0 ? Math.max((ntlChartW - ntlPad.l - ntlPad.r) / total - 1, 1) : 2
}

function cloudBarX(dashId, i) {
  const total = getAllDays(dashId).length
  const w = (ntlChartW - ntlPad.l - ntlPad.r) / total
  return ntlPad.l + i * w + 0.5
}

function cloudSplitX(dashId) {
  const pre = getPreDays(dashId).length
  const total = getAllDays(dashId).length
  return total > 0 ? ntlPad.l + (pre / total) * (ntlChartW - ntlPad.l - ntlPad.r) : ntlPad.l
}

const allSections = [
  { id: 'overview',     num: '01', title: 'Project Overview', tags: ['VIIRS VNP46A2'] },
  { id: 'data',         num: '02', title: 'Data Collection & Processing', tags: ['VNP46A2', 'OSM', 'EAGLE-I'] },
  { id: 'eda',          num: '03', title: 'Exploratory Data Analysis', tags: ['Resilience Ratio', 'Floor Effect'] },
  { id: 'interpretive', num: '04', title: 'Interpretive Modeling', tags: ['OLS', 'MixedLM', 'Logit', 'Cox'] },
  { id: 'features',     num: '05', title: 'Feature Engineering', tags: ['17 features'] },
  { id: 'models',       num: '06', title: 'Predictive Models & Probability Maps', tags: ['RF + XGB', 'LOEO', '4 Variants'] },
  { id: 'stage3',       num: '07', title: 'Zip-Code Analysis', tags: ['EAGLE-I', 'Spatial Regression'] },
  { id: 'conclusions',  num: '08', title: 'Conclusions & Future Work', tags: ['Limitations', 'Future Directions'] },
  { id: 'web',          num: '09', title: 'Dashboard Development', tags: ['Vue 3', 'MapLibre'] },
  { id: 'repro',        num: '10', title: 'Reproducibility', tags: ['Open data'] },
]

const sectionData = computed(() => allSections.find(s => s.id === sectionId.value))

// Cloud fraction chart helpers
function getCfDays(dashId) { return cloudFrac.value?.[dashId]?.days ?? [] }
function cfBarW(dashId) {
  const total = getCfDays(dashId).length
  return total > 0 ? Math.max((ntlChartW - ntlPad.l - ntlPad.r) / total - 1, 1) : 2
}
function cfBarX(dashId, i) {
  const total = getCfDays(dashId).length
  const w = (ntlChartW - ntlPad.l - ntlPad.r) / total
  return ntlPad.l + i * w + 0.5
}
function cfSplitX(dashId) {
  const days = getCfDays(dashId)
  const preCount = days.filter(d => d.period === 'pre').length
  const total = days.length
  return total > 0 ? ntlPad.l + (preCount / total) * (ntlChartW - ntlPad.l - ntlPad.r) : ntlPad.l
}

// Events sorted chronologically by year
const sortedEvents = computed(() => [...EVENTS].sort((a, b) => a.year - b.year))

// EDA chart helpers
const floorCities = {
  large: 'San Juan, Miami, New Orleans',
  medium: 'Fort Myers, Hatay',
  small: 'Lake Charles, Panama City, Charlotte Harbor',
}

const floorGroups = [
  { key: 'large',  label: 'Large',  cities: 'San Juan, Miami, New Orleans' },
  { key: 'medium', label: 'Medium', cities: 'Fort Myers, Hatay' },
  { key: 'small',  label: 'Small',  cities: 'Lake Charles, Panama City, Charlotte Harbor' },
]

function floorVal(citySize, inBuffer) {
  if (!edaStats.value) return 0
  const d = edaStats.value.floor.find(f => f.city_size === citySize && f.in_buffer === inBuffer)
  return d ? d.mean_pre_ntl : 0
}

const sortedEdaEvents = computed(() => {
  if (!edaStats.value) return []
  return [...edaStats.value.events].sort((a, b) => b.ra - a.ra)
})

// ── Sub-section TOC per detail page ──
const SUB_SECTIONS = {
  overview: [
    { id: 'sec-1-1', label: '1.1 The Data Gap' },
    { id: 'sec-1-2', label: '1.2 Can Satellites Help?' },
    { id: 'sec-1-3', label: '1.3 Approach' },
    { id: 'sec-1-4', label: '1.4 Study Areas' },
    { id: 'sec-1-5', label: '1.5 Research Questions' },
  ],
  data: [
    { id: 'sec-2-1', label: '2.1 NASA Black Marble' },
    { id: 'sec-2-2', label: '2.2 EAGLE-I Outages' },
    { id: 'sec-2-3', label: '2.3 Disaster in NTL' },
    { id: 'sec-2-4', label: '2.4 Cloud & QC' },
    { id: 'sec-2-5', label: '2.5 GEE Acquisition' },
    { id: 'sec-2-6', label: '2.6 Generator Permits' },
    { id: 'sec-2-7', label: '2.7 OSM Facilities' },
    { id: 'sec-2-8', label: '2.8 Data Quality' },
  ],
  eda: [
    { id: 'sec-4-1', label: '3.1 Key Definitions' },
    { id: 'sec-4-2', label: '3.2 Buffer vs Non-Buffer' },
    { id: 'sec-4-3', label: '3.3 Floor Effect' },
    { id: 'sec-4-4', label: '3.4 Facility Types' },
    { id: 'sec-4-5', label: '3.5 City Size Effects' },
    { id: 'sec-4-6', label: '3.6 Key Findings' },
  ],
  interpretive: [
    { id: 'sec-5-1', label: '4.1 Why Four Models?' },
    { id: 'sec-5-2', label: '4.2 Specification' },
    { id: 'sec-5-3', label: '4.3 OLS' },
    { id: 'sec-5-4', label: '4.4 MixedLM' },
    { id: 'sec-5-5', label: '4.5 Logistic' },
    { id: 'sec-5-6', label: '4.6 Cox PH' },
    { id: 'sec-5-7', label: '4.7 Land-Use Confound' },
  ],
  features: [
    { id: 'sec-6-floor', label: '5.1 Floor Effect' },
    { id: 'sec-6-features', label: '5.2 Feature Set' },
  ],
  models: [
    { id: 'sec-5-intro', label: '6.1 Interpretation → Prediction' },
    { id: 'sec-5-algo', label: '6.2 Model A — Full' },
    { id: 'sec-5-loeo', label: '6.6 LOEO Design' },
  ],
  stage3: [
    { id: 'sec-8-1', label: '7.1 Research Question' },
    { id: 'sec-8-2', label: '7.2 Data Sources' },
    { id: 'sec-8-3', label: '7.3 Sample' },
    { id: 'sec-8-4', label: '7.4 Models & Results' },
  ],
  conclusions: [
    { id: 'sec-c-1', label: 'Conclusions' },
    { id: 'sec-c-2', label: 'Limitations' },
    { id: 'sec-c-3', label: 'Future Directions' },
  ],
  web: [
    { id: 'sec-8-1', label: '9.1 Architecture' },
    { id: 'sec-8-2', label: '9.2 Map Engine' },
    { id: 'sec-8-3', label: '9.3 Responsive Design' },
    { id: 'sec-8-4', label: '9.4 Deployment' },
  ],
}

const subSections = computed(() => SUB_SECTIONS[sectionId.value] ?? [])
const activeTocId = ref('')

function tocScrollTo(id) {
  const el = document.getElementById(id)
  if (!el) return
  el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

// Scroll reveal for detail pages
let revealObs
function setupReveal() {
  if (revealObs) revealObs.disconnect()
  revealObs = new IntersectionObserver(
    entries => entries.forEach(e => e.target.classList.toggle('visible', e.isIntersecting)),
    { threshold: 0.1 }
  )
  setTimeout(() => document.querySelectorAll('.detail-page .reveal').forEach(el => revealObs.observe(el)), 200)
}

// Intersection observer for TOC active state
let tocObserver
onMounted(() => {
  setupTocObserver()
  setupReveal()
})

function setupTocObserver() {
  if (tocObserver) tocObserver.disconnect()
  const ids = subSections.value.map(s => s.id)
  if (!ids.length) return
  tocObserver = new IntersectionObserver(
    entries => entries.forEach(e => { if (e.isIntersecting) activeTocId.value = e.target.id }),
    { rootMargin: '-10% 0px -70% 0px' }
  )
  // Wait for DOM
  setTimeout(() => {
    ids.forEach(id => { const el = document.getElementById(id); if (el) tocObserver.observe(el) })
  }, 200)
}
const currentIndex = computed(() => allSections.findIndex(s => s.id === sectionId.value))
const prevSection = computed(() => currentIndex.value > 0 ? allSections[currentIndex.value - 1] : null)
const nextSection = computed(() => currentIndex.value < allSections.length - 1 ? allSections[currentIndex.value + 1] : null)

const features17 = [
  { name: 'drop_magnitude',       desc: 'Clipped NTL drop (outage signal)' },
  { name: 'delta_ntl',            desc: 'Raw relative NTL change' },
  { name: 'log_pre_ntl',          desc: 'Log pre-disaster brightness' },
  { name: 'log_post_ntl',         desc: 'Log post-disaster brightness' },
  { name: 'log_city_pre_mean',    desc: 'City-level log mean NTL' },
  { name: 'ntl_relative',         desc: 'Pixel brightness / city mean' },
  { name: 'log_dist',             desc: 'Log distance to nearest facility' },
  { name: 'near_fire_station',    desc: 'Binary: nearest = fire station' },
  { name: 'near_police',          desc: 'Binary: nearest = police' },
  { name: 'near_excluded',        desc: 'Binary: nearest = excluded type' },
  { name: 'fac_group',            desc: 'Facility group ordinal (1/2/3)' },
  { name: 'city_size_code',       desc: 'large=0, medium=1, small=2' },
  { name: 'is_hurricane',         desc: 'Disaster type flag' },
  { name: 'is_earthquake',        desc: 'Disaster type flag' },
  { name: 'ntl_x_group',         desc: 'log_pre × fac_group interaction' },
  { name: 'below_city_median',    desc: 'Pixel below city median NTL' },
  { name: 'below_median_x_group', desc: 'below_median × fac_group' },
]

const modelVariants = [
  { id: 'Model A', name: 'Pre + Post NTL',       desc: 'Full feature set. Primary model. Probability map correlates with baseline brightness.' },
  { id: 'Model B', name: 'Post-Disaster Only',   desc: 'No pre-NTL features. Avoids baseline bias in probability maps.' },
  { id: 'Model C', name: 'A + Building Coverage', desc: 'Adds OSM building footprint coverage per pixel. Addresses 500m mixed-pixel problem.' },
]
</script>

<style scoped>
.reveal {
  opacity: 0;
  transform: translateY(30px);
  transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1), transform 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}
.reveal.visible {
  opacity: 1;
  transform: translateY(0);
}
.detail-page {
  min-height: calc(100vh - var(--nav-h));
  background: transparent;
  display: flex;
  max-width: 1100px;
  margin: 0 auto;
  gap: 0;
}
.detail-inner {
  flex: 1;
  min-width: 0;
  max-width: 820px;
  padding: 32px 32px 80px;
  background: rgba(3,13,26,0.6);
  backdrop-filter: blur(4px);
  border-left: 1px solid rgba(18,42,69,0.3);
  border-right: 1px solid rgba(18,42,69,0.3);
}

/* ── Sidebar TOC ── */
.detail-toc {
  position: sticky;
  top: calc(var(--nav-h) + 24px);
  align-self: flex-start;
  width: 200px;
  flex-shrink: 0;
  padding: 24px 16px 24px 24px;
}
.detail-toc__title {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 700;
  color: var(--text-bright);
  margin-bottom: 12px;
  letter-spacing: 0.02em;
}
.detail-toc__list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.detail-toc__link {
  display: block;
  padding: 5px 10px;
  border-radius: var(--radius);
  border-left: 2px solid transparent;
  font-size: 12px;
  color: var(--text-muted);
  text-decoration: none;
  cursor: pointer;
  transition: all var(--t-fast);
  line-height: 1.4;
}
.detail-toc__link:hover {
  color: var(--text-bright);
  background: var(--bg-3);
}
.detail-toc__link.active {
  color: var(--cyan);
  background: var(--cyan-dim);
  border-left-color: var(--cyan);
}

/* ── Takeaway box ── */
.takeaway {
  background: linear-gradient(135deg, rgba(255,170,0,0.08), rgba(255,100,0,0.05));
  border: 1px solid rgba(255,170,0,0.25);
  border-left: 4px solid #ffaa00;
  border-radius: var(--radius-lg);
  padding: 24px 28px;
  margin: 32px 0 16px;
}
.takeaway__label {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.16em;
  color: #ffaa00;
  margin-bottom: 12px;
}
.takeaway__text {
  font-size: 17px;
  line-height: 1.8;
  color: var(--text-bright);
}
.takeaway__text strong {
  color: #ffaa00;
}
.takeaway__text em {
  color: var(--cyan);
  font-style: normal;
}

/* h2 scroll margin for TOC */
.detail-content h2[id] {
  scroll-margin-top: calc(var(--nav-h) + 24px);
}

/* Back link */
.back-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
  color: var(--cyan);
  text-decoration: none;
  margin-bottom: 24px;
  transition: all var(--t-fast);
}
.back-link:hover { color: var(--text-bright); }
.back-arrow {
  transition: transform var(--t-fast);
}
.back-link:hover .back-arrow { transform: translateX(-3px); }

/* Header */
.detail-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 28px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.detail-header h1 { font-size: 26px; font-weight: 700; }
.dim { font-size: 12px; color: var(--text-dim); }

/* Content styling */
.detail-content h2 { font-size: 18px; font-weight: 600; color: var(--text-bright); margin: 28px 0 12px; }
.detail-content h3 { font-size: 15px; font-weight: 600; color: var(--text-bright); margin: 24px 0 10px; }
.detail-content p { font-size: 14px; color: var(--text); line-height: 1.75; margin-bottom: 12px; }
.detail-content strong { color: var(--text-bright); }
.detail-content em { color: var(--cyan); font-style: normal; }
code {
  font-family: var(--font-mono); font-size: 12px;
  background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.2);
  border-radius: 3px; padding: 2px 6px; color: var(--green);
  white-space: nowrap;
}
.inline-link { color: var(--cyan); text-decoration: none; }
.inline-link:hover { text-decoration: underline; }

.detail-list {
  list-style: none;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 12px 0;
}
.detail-list li {
  font-size: 13px;
  color: var(--text);
  line-height: 1.6;
  padding-left: 18px;
  position: relative;
}
.detail-list li::before {
  content: '▸';
  position: absolute;
  left: 0;
  color: var(--cyan);
}

/* Callouts */
.callout { display: flex; gap: 12px; padding: 12px 16px; border-radius: var(--radius-lg); margin: 12px 0; font-size: 13px; line-height: 1.6; }
.callout--cyan  { background: var(--cyan-dim);  border: 1px solid rgba(0,212,255,.15); color: var(--text); }
.callout--amber { background: var(--amber-dim); border: 1px solid rgba(255,170,0,.15); color: var(--text); }
.callout--green { background: var(--green-dim); border: 1px solid rgba(0,229,160,.15); color: var(--text); }

/* Tables */
.data-table { margin: 12px 0; overflow-x: auto; }
.data-table table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
  font-family: var(--font-head); font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-dim);
  text-align: left; padding: 8px 12px;
  background: var(--bg-2); border-bottom: 1px solid var(--border);
}
td { padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-3); }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }

/* Formula */
.formula-block {
  background: var(--bg-2); border: 1px solid var(--border);
  border-left: 3px solid var(--cyan);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 14px 18px; margin: 12px 0;
}
.formula { font-family: var(--font-mono); font-size: 14px; color: var(--text-bright); }
.formula__caption { margin-top: 4px; font-size: 11px; color: var(--text-muted); }

/* Code block */
.code-block { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; margin: 12px 0; }
.code-block__header { padding: 7px 14px; background: var(--bg-3); border-bottom: 1px solid var(--border); font-size: 11px; color: var(--text-muted); }
.code-block pre { padding: 14px; overflow-x: auto; margin: 0; }
.code-block code { font-size: 12px; background: none; border: none; padding: 0; color: var(--text); line-height: 1.7; display: block; white-space: pre; }

/* Feature grid */
.feature-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; margin: 10px 0; }
.feature-item { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 9px 11px; }
.feature-item__name { font-size: 11px; color: var(--cyan); margin-bottom: 3px; }
.feature-item__desc { font-size: 11px; color: var(--text-muted); line-height: 1.4; }

/* Model cards */
.model-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0; }
.model-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 14px; }
.model-card__label { font-size: 10px; color: var(--cyan); letter-spacing: 0.12em; margin-bottom: 4px; font-family: var(--font-mono); }
.model-card__name { font-family: var(--font-head); font-size: 13px; font-weight: 600; color: var(--text-bright); margin-bottom: 5px; }
.model-card__desc { font-size: 12px; color: var(--text-muted); line-height: 1.5; }

/* Bottom navigation */
.detail-nav {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}
.detail-nav__link {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 16px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  text-decoration: none;
  transition: all var(--t-fast);
  min-width: 140px;
}
.detail-nav__link:hover {
  border-color: var(--cyan);
  background: var(--bg-3);
}
.detail-nav__link--next { text-align: right; margin-left: auto; }
.detail-nav__dir {
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--cyan);
}
.detail-nav__name {
  font-size: 13px;
  color: var(--text-bright);
  font-weight: 500;
}

.not-found {
  text-align: center;
  padding: 80px 0;
  color: var(--text-muted);
}

/* EDA chart cards */
.eda-chart-card {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
  margin: 16px 0;
}

/* NTL Animation Player */
.ntl-player {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
  margin: 16px 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.ntl-player__panels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.ntl-player__panel {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.ntl-player__label {
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.ntl-player__img {
  width: 100%;
  height: auto;
  border-radius: var(--radius);
  background: #000;
  image-rendering: pixelated;
}
.ntl-player__info {
  display: flex;
  align-items: center;
  gap: 12px;
}
.ntl-player__date {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-bright);
}
.ntl-player__phase {
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  padding: 3px 8px;
  border-radius: 3px;
}
.ntl-player__phase.pre {
  background: rgba(0,229,160,0.15);
  color: var(--green);
}
.ntl-player__phase.post {
  background: rgba(255,107,107,0.15);
  color: #ff6b6b;
}
.ntl-player__controls {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ntl-player__btn {
  width: 32px;
  height: 32px;
  border-radius: var(--radius);
  background: var(--bg-3);
  border: 1px solid var(--border);
  color: var(--text);
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--t-fast);
}
.ntl-player__btn:hover { border-color: var(--cyan); color: var(--cyan); }
.ntl-player__btn--play { width: 40px; font-size: 16px; }
.ntl-player__slider {
  flex: 1;
  height: 4px;
  appearance: none;
  background: var(--bg-4, var(--border));
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}
.ntl-player__slider::-webkit-slider-thumb {
  appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--cyan);
  cursor: pointer;
}
.ntl-player__legends {
  display: flex;
  gap: 24px;
}
.ntl-player__legend {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.legend-bar {
  height: 8px;
  width: 120px;
  border-radius: 3px;
}
.legend-bar--hot {
  background: linear-gradient(90deg, #000, #a00, #f60, #ff0, #fff);
}
.legend-bar--div {
  background: linear-gradient(90deg, #00f, #008, #000, #800, #f00);
}
.legend-labels {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--text-dim);
  width: 120px;
}

/* Collapsible sections */
.collapsible {
  position: relative;
  margin: 16px 0;
}
.collapsible--code .collapsible__content {
  max-height: 170px;
}
.collapsible__content {
  max-height: 220px;
  overflow: hidden;
  transition: max-height 0.4s ease;
}
.collapsible.expanded .collapsible__content {
  max-height: 8000px;
}
.collapsible__fade {
  position: absolute;
  bottom: 36px;
  left: 0; right: 0;
  height: 120px;
  background: linear-gradient(to bottom, rgba(3,13,26,0) 0%, rgba(3,13,26,0.95) 85%);
  pointer-events: none;
  z-index: 1;
}
.collapsible__toggle {
  display: block;
  width: 100%;
  padding: 10px 0;
  background: none;
  border: none;
  border-top: 1px solid var(--border);
  cursor: pointer;
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--cyan);
  text-align: center;
  transition: all var(--t-fast);
  position: relative;
  z-index: 2;
}
.collapsible__toggle:hover {
  color: var(--text-bright);
  background: var(--bg-3);
}

/* NTL / Cloud chart cards */
.ntl-charts-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin: 16px 0;
}
.ntl-chart-card {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 12px;
  overflow: hidden;
}
.ntl-chart-card__header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 13px;
}
.ntl-svg {
  width: 100%;
  height: auto;
  display: block;
}

/* Feature link (map/docs cross-link) */
.feature-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin: 24px 0;
  padding: 16px 20px;
  background: var(--cyan-dim);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: var(--radius-lg);
  text-decoration: none;
  transition: all var(--t-fast);
}
.feature-link:hover {
  border-color: var(--cyan);
  background: rgba(0,212,255,0.12);
}
.feature-link__text {
  font-size: 14px;
  color: var(--text-bright);
}
.feature-link__cta {
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 600;
  color: var(--cyan);
  white-space: nowrap;
}

/* Chart modal */
.chart-modal {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(0,0,0,0.7);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  animation: fadeIn 0.15s ease;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.chart-modal__card {
  background: var(--bg-2);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-lg);
  padding: 24px 28px;
  max-width: 780px;
  width: 100%;
  position: relative;
}
.chart-modal__close {
  position: absolute;
  top: 12px; right: 16px;
  background: none; border: none;
  color: var(--text-muted); font-size: 22px;
  cursor: pointer;
}
.chart-modal__close:hover { color: var(--text-bright); }
.chart-modal__header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  font-size: 15px;
}

@media (max-width: 900px) {
  .detail-page { flex-direction: column; }
  .detail-toc { position: static; width: 100%; padding: 16px 16px 0; }
  .detail-toc__list { flex-direction: row; flex-wrap: wrap; gap: 4px; }
  .detail-inner { padding: 24px 16px 60px; }
  .model-cards { grid-template-columns: 1fr; }
  .detail-nav { flex-direction: column; }
  .ntl-charts-grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 600px) {
  .detail-inner { padding: 16px 12px 60px; max-width: 100vw; overflow-x: hidden; }
  .detail-header h1 { font-size: 20px; }
  .detail-content h2 { font-size: 16px; }
  .detail-content p { font-size: 13px; }
  .takeaway { padding: 16px 14px; }
  .takeaway__text { font-size: 14px; }
  .formula-block { padding: 10px 12px; }
  .formula { font-size: 12px; word-break: break-all; }
  .ntl-player__panels { grid-template-columns: 1fr; }
  .ntl-player__legends { flex-direction: column; gap: 12px; }
  .data-table { margin: 8px -12px; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .data-table table { font-size: 11px; min-width: 400px; }
  th, td { padding: 6px 8px; }
  .feature-grid { grid-template-columns: 1fr; }
  .code-block pre { padding: 10px; }
  .code-block code { font-size: 11px; }
  .callout { flex-direction: column; gap: 6px; }
  .detail-toc__link { font-size: 11px; padding: 4px 8px; }
  .ntl-charts-grid { grid-template-columns: 1fr; }
  .feature-link { flex-direction: column; gap: 8px; }
  .chart-modal__card { padding: 16px; }
}
</style>
